# Verus artifact audit guide for Jun

**Purpose**: onboard Jun to the paper's Verus artifact so he can audit
efficiently. Written by Claude (AI drafter). Wei has read this and
signed off on the structure; Jun is the auditor of record.

**Status** (updated 2026-09-11):
- **220 unique verified proof functions, 0 errors** across **20 Verus source
  files**. This counts shared modules once. Standalone command totals must not
  be added together because the core, kernel, schedule, and deadlock modules
  are re-verified when imported by another crate root.
- The Tier-3 bridge files contain no new `#[verifier::external_body]`.
  The public block theorem now imports and invokes Ex04, Ex05, Ex06, and Ex09
  contracts to establish the GQA-to-MoE seam, routing, ownership, and fused-row
  obligations. Ex04 reuses Ex01's all-reduce postcondition. The imported
  component files retain their documented low-level model-to-source axioms.
- All files typecheck against Verus 0.2025.07.12.0b6f3cb.
- Every proof has been drafted by Claude in this session; Jun's audit
  is the human-verification step of the paper's methodology contribution.

## 1. Audit protocol (read this first)

The paper's methodology claim is: **contracts and proofs are AI-drafted
by Claude and audited by a Verus expert (Jun) — the audit is the
correspondence-check between Verus formalism and Python implementation.**

Concretely, for each file, Jun should answer five questions:

1. **Does it typecheck?** Run `verus <file>.rs` (or
   `verus --crate-type=lib <file>.rs` for the top-level `verus/` files).
   Confirm `verification results:: N verified, 0 errors`.
2. **Are the axioms sound?** Each `#[verifier::external_body]` block is
   a trust point. Read it and confirm the ensures-clause faithfully
   mirrors the corresponding Python code or is a well-known
   mathematical fact.
3. **Are the contracts non-decorative?** For each claimed property,
   ask: "if I delete the semantic clauses and keep only the shape
   clauses, does the proof still go through?" If yes, the semantic
   clauses are decoration and the contract needs strengthening.
4. **Are proofs honest?** No `admit()` calls in the final proof. Every
   `#[verifier::external_body]` maps to either a Python-correspondence
   axiom or a well-known math fact.
5. **Are theorem preconditions discharged?** A conditional theorem may be
   perfectly valid while proving nothing about the Python program yet. For
   every refinement/schedule/runtime contract in `requires`, identify a
   concrete upstream lemma. If the upstream stub only says `ensures true`,
   record the contract as undischarged.

## 2. File-by-file directory

### 2.1 Per-component proofs (Tier 1)

Each file lives at `bootcamp/exNN_<name>/verification/<component>.rs`
alongside a `PROPERTIES.md` that states the semantic spec in prose and
math. Read `PROPERTIES.md` first, then the `.rs`.

| Exercise | File | Verified | What it proves | Key axioms trusted |
|----------|------|---------:|----------------|--------------------|
| Ex01 linear_tp (col) | `column_parallel.rs` | 12 | C1-C4 for ColumnParallelLinear | AXIOM_M1 (matmul splits over out-dim) |
| Ex01 linear_tp (row) | `row_parallel.rs` | 13 | R1-R4 for RowParallelLinear + R4b (Replicated after all-reduce) | AXIOM_M2, `axiom_all_reduce_produces_replicated` |
| Ex02 mlp_tp | `mlp_tp.rs` | 19 | M1-M3 merged column + exact tp=2 T1-T3 SwiGLU block composition | AXIOM_M1, meaningful reconstruction-conditioned AXIOM_M2, AXIOM_S1 |
| Ex03 mha_tp | `mha_tp.rs` | 22 | Q1-Q4 QKV projection/split + exact tp=2 A1-A2 attention block composition | AXIOM_M1, reconstruction-conditioned AXIOM_M2, AXIOM_ATTN_HEAD_LOCAL |
| Ex04 gqa_tp | `gqa_tp.rs` | 12 | G1-G5 GQA + R1 KV-replication invariants, including replicated output projection | AXIOM_RI1 (repeat_interleave), Ex01 all-reduce replication axiom |
| Ex05 moe_baseline | `moe_baseline.rs` | 24 | RT1-RT3 routing invariants + RT5 partition-size correctness (catches v8/v10) + unconditional conservation + RT4 permutation bijection; exact E1/E2 live in the shared Tier-3 model | AXIOM_SOFTMAX_SUM, AXIOM_ARGSORT_INVERSE |
| Ex06_ep_pure | `ep_pure.rs` | 8 | EP1-EP4 expert-partition + dispatch symmetry | AXIOM_A2A_COUNT_SYMMETRIC |
| Ex06_ep (lean) | `lean.rs` | 7 | L2 disjointness + L2-covers + L3/L5; exact L4 lives in the shared Tier-3 model | AXIOM_ALLREDUCE_SUM_REPLICATED |
| Ex07 tp_ep_hybrid | `hybrid.rs` | 3 | H2 striping determinism; exact H6 lives in the shared Tier-3 model | tensor_on opaque |
| Ex09 fused_moe (contract) | `fused_moe.rs` | 2 | F1 offset coverage + F3 empty-expert; exact F4 lives in the shared Tier-3 model | expert_apply opaque |
| Ex09 fused_moe (DSL) | `fused_kernel_dsl.rs` | 24 | K1 tile coverage/disjointness, exact K2 reduction, grouped-GEMM K3, and three-GEMM fused F4 | none; source-to-DSL correspondence remains audited |
| **Subtotal** | | **146** | | |

The Ex01 row file has 13 unique local verification items; its standalone
command now reports 29 because it imports the 16-item shared composition model
to use the same distributed tensor observation as Tier 3.

The Ex04 file has 12 unique local verification items; its standalone command
reports 57 because G5 imports Ex01 row and both modules import the shared
composition model. These imported results must not be counted again.

**Ex01 detail** — the paper's exemplar per-component proof. Read
[`bootcamp/ex01_linear_tp/verification/PROPERTIES.md`](../bootcamp/ex01_linear_tp/verification/PROPERTIES.md)
first — it argues the three-tool structure (Verus + Dafny + Z3) that
the paper's methodology section names. The Verus file follows this
pattern exactly, and every subsequent per-component .rs mirrors it.

**Ex02 detail** — first composition proof. Ships with an
[`anecdote.txt`](../bootcamp/ex02_mlp_tp/verification/anecdote.txt) —
the audit-cycle log for this file. Verus rejected Claude's initial
draft twice (missing well-formedness scaffolding + missing div_mod
lemma). Read anecdote.txt as evidence that Verus catches real
proof-drafting errors; this is the concrete data point for
Contribution 3.

### 2.2 Shared framework (Tier 2)

There are now two distinct shared layers:

- `verus/axiom_base.rs` (4 verified lemmas) is the legacy distributed-state
  framework used by older component files. Its equality relation is exact;
  current Tier-3 composition uses the separate shared model below.
- `verus/composition_core.rs` (16 verified proofs) is the Tier-3 source of
  truth. In addition to exact `Tensor` equality and the meta-theorem, it
  defines the shared `(token, top-k slot)` work model, exact expert
  contributions, routing-permutation semantics, fused row contract, and the
  permutation/work-conservation proofs.

The Tier-3 model proves exact equality.

### 2.3 Composition + safety theorems (Tier 3)

Eight files at the repo-level `verus/` directory. The two small crate-root
files re-export shared proof modules so they can also be checked standalone.

| File | Command output | Unique local proofs | What it proves |
|------|---------------:|--------------------:|----------------|
| `composition_core.rs` | 16 | 16 | Exact equality, work semantics, permutation invariance, fused-row refinement, and work conservation |
| `kernel_refinement.rs` | imported | 6 | E1, E2, F4 and Naive/Permuted/Fused equivalences |
| `schedule_refinement.rs` | imported | 18 | Expert ownership, rank-local partial outputs, elementwise all-reduce SUM, explicit all-to-all dispatch/compute/combine round trip, schedule swap, and per-schedule kernel swap |
| `component_integration.rs` | imported | 8 | Bridges Ex04/Ex01 replication through residual-add and RMSNorm, Ex05 argsort, Ex06 expert partition, and the proved Ex09 DSL execution into the shared Tier-3 predicates |
| `lean_equiv_hybrid_dp1.rs` | 40 | 0 | Standalone crate root importing core + kernel + schedule modules (historical filename) |
| `naive_equiv_fused_moe.rs` | 22 | 0 | Standalone crate root importing core + kernel modules |
| `composition_theorem.rs` | 184 | 10 | Component-connected block composition + work, deadlock, and atomic-scatter safety; imports shared modules and relevant Tier-1 proofs |
| `deadlock_free.rs` | 12 | 12 | Transition-system deadlock proof + concrete DP=1 all-reduce and all-to-all schedules |
| **Unique subtotal** | | **70** | Shared modules counted once; imported Tier-1 proofs remain counted in the Tier-1 subtotal |

**The meta-theorem** is defined in `composition_core.rs`. E1/E2/F4 prove the
kernel refinements from canonical work, routing permutation, and the row-level
fused contract. L6/H6 then prove the schedule refinements. The block theorem
invokes those results directly; it does not assume refinement or swap facts.
The public block theorem first invokes `component_integration.rs` to derive the
shared invariants from Tier-1 contracts. It requires `valid_dp1_topology`,
whose definition includes `dp_size == 1`; it makes no equivalence claim for
the other `(dp,tp)` points.

## 3. Grand total

**220 unique verified proof functions, 0 errors, 20 source files**
(per-component: 146; `axiom_base.rs`: 4; current composition cluster: 70).
Re-verified 2026-09-11. Shared imports are counted once.

## 4. Property matrix — what's actually proved

### Functional equivalence properties

| Claim | Where | Lemma |
|-------|-------|-------|
| ColumnParallelLinear forward = unsharded matmul | Ex01 col | `c4_forward_correctness_tp2` |
| RowParallelLinear sum-of-partials = unsharded matmul | Ex01 row | `r4_forward_correctness_tp2` |
| RowParallelLinear output is Replicated on tp_group | Ex01 row | `r4b_output_replicated_after_all_reduce` |
| Merged column-parallel = unsharded merged matmul | Ex02 | `m3_merged_forward_correctness_tp2` |
| TP=2 SwiGLU MLP block = unsharded MLP | Ex02 | `t3_block_correctness_tp2` |
| TP=2 Q/K/V projection shards reconstruct unsharded projections | Ex03 | `q3_qkv_forward_correctness_tp2` |
| Rank-local packed QKV output splits into the three shard projections | Ex03 | `q4_three_way_split` |
| TP=2 MHA block = unsharded MHA | Ex03 | `a2_block_correctness_tp2` |
| GQA weight_loader post-condition | Ex04 | `g4_gqa_weight_loader_postcondition` |
| GQA output projection is replicated after all-reduce | Ex04 | `g5_gqa_output_replicated_after_output_projection` |
| Ex05 argsort establishes shared routing consistency | `component_integration.rs` | `ex05_establishes_routing_consistent` |
| Ex06 contiguous partition establishes shared expert ownership | `component_integration.rs` | `ex06_establishes_expert_partitioned` |
| GQA output replication survives residual-add/RMSNorm and establishes MoE input replication | `component_integration.rs` | `ex04_establishes_replicated_post_attention_input` |
| Ex09 DSL equals the Python mathematical expert reference | `fused_kernel_dsl.rs` | `k2_k_reduce_correctness`, `k3_grouped_matmul_dsl_equals_reference`, `f4_fused_moe_dsl_equals_python_reference` |
| Ex09 DSL execution establishes the component row postcondition | `component_integration.rs` | `ex09_dsl_execution_establishes_postcondition`, `ex09_establishes_fused_rows_correct` |
| Naive MoE refines MoE_spec | `kernel_refinement.rs` | `e1_naive_refines_spec` |
| Permuted MoE refines MoE_spec | `kernel_refinement.rs` | `e2_permuted_refines_spec`, from `RoutingConsistent` |
| Fused rows refine MoE_spec | `kernel_refinement.rs` | `f4_fused_refines_spec`, from routing + pointwise `fused_rows_correct` |
| **META**: A = S ∧ B = S ⟹ A = B | `composition_core.rs` | `theorem_shared_spec_implies_equiv` |
| All-to-all dispatch/compute/combine round trip | `schedule_refinement.rs` | `theorem_all_to_all_dispatch_compute_combine_round_trip`; records move to their expert owner, retain identity/value through compute, and return to their token source without loss or duplication |
| All-to-all ≡ all-reduce under DP=1 | `schedule_refinement.rs` | `theorem_all_to_all_equiv_all_reduce_dp1`; `theorem_rank_local_partials_sum_to_moe_spec` proves the replicated-EP partial-sum identity, while the explicit round-trip theorem supplies H6 |
| Permuted ≡ Fused Triton | `kernel_refinement.rs` | `theorem_permuted_equiv_fused` |
| Naive ≡ Fused Triton | `kernel_refinement.rs` | `theorem_naive_equiv_fused` |
| Naive ≡ Permuted | `kernel_refinement.rs` | `corollary_naive_equiv_permuted` |
| Block-level 2×2 grid equivalence | `composition_theorem.rs` §1 | `corollary_block_variants_from_components`; invokes Ex04/Ex05/Ex06/Ex09 bridges before the concrete refinements; Ex04 invokes Ex01's all-reduce contract |

### Global safety goals — the paper's four goals, and where each is PROVED

The block-scope theorems in `composition_theorem.rs` §2 contain no
`external_body`. Work conservation is derived from routing permutation;
atomic-scatter safety follows from the modeled `AtomicIndexAdd` primitive;
deadlock freedom imports the concrete transition-system proof.

**Goal 1 — Completeness** (every work item assigned + executed; $\bigcup_r W_r = W$)
- Block scope: `theorem_token_conservation_from_components` (§2), which calls
  Ex05's argsort proof to establish the routing permutation.
- A work item is `(token, top_k_slot)`. The theorem proves every required item
  has execution multiplicity exactly one.
- Proved per-component:
  - Ex05 `rt1_token_conservation_from_assignment` (`moe_baseline.rs`) — offset
    endpoint == token count, **unconditional** (the previously-assumed
    sum premise is discharged by `lemma_bincount_sums_to_len`).
  - Ex06_ep_pure `ep2_expert_coverage`; Ex06_ep/lean `l2_local_mask_covers`.
  - Ex09 `f1_offsets_cover_range` (`fused_moe.rs`), `k1b_n_axis_covers`
    (`fused_kernel_dsl.rs`).

**Goal 2 — Disjointness** (no work item done twice; $W_r \cap W_s = \varnothing$)
- Block scope: `work_disjoint` from the same theorem. It proves every work
  item's execution multiplicity is at most one. `no_spurious_work` separately
  excludes execution of non-required items.
- Proved per-component:
  - Ex05 `rt5_block_size_matches_count` (`moe_baseline.rs`) — each expert
    block has exactly its true token count, so the blocks partition with no
    overlap; `rt5_wrong_size_implies_wrong_offsets` is the corruption-detection
    contrapositive (rejects the v8/v10 offset bugs).
  - Ex06_ep_pure `ep1_expert_sharding_disjoint`; Ex06_ep/lean `l2_local_mask_disjoint`.
  - Ex01 `c2_sharding_disjoint` (`column_parallel.rs`), `r2_sharding_disjoint_dim1` (`row_parallel.rs`).
  - Ex09 `k1a_m_axis_disjoint`, `k1c_n_axis_disjoint` (fused tiles disjoint).

**Goal 3 — Data-race freedom** (conflicting scatter writes are atomic)
- Block scope: `theorem_data_race_free` (§2), with the abstract scatter
  primitive explicitly defined as `AtomicIndexAdd`.
- Different top-k slots intentionally write the same `(token, feature)`.
  `ScatterWrite.occurrence` also represents accidental duplicate execution.
  The theorem proves every pair of conflicting active writes is atomic; it
  does **not** claim a unique writer or deterministic reduction order.
- Ex05 RT4 proves a bijection over flattened routing records, not uniqueness
  of the final output-row destination.

**Goal 4 — Deadlock freedom** (every reachable non-final state can progress)
- `deadlock_free.rs` models per-rank collective traces, program counters,
  enabled transitions, frontier closure, and reachability.
- `lemma_all_reduce_dp1_schedule_well_formed` proves the concrete two-phase
  all-reduce trace.
- `lemma_all_to_all_dp1_schedule_well_formed` proves the concrete four-phase
  attention + count/dispatch/combine all-to-all trace.
- `composition_theorem.rs::theorem_deadlock_free` selects the schedule and
  invokes the corresponding DP=1 execution theorem.
- This is source-level absence of a deadlocked reachable state, not an
  unconditional runtime termination theorem. Failure-free NCCL completion and
  fair scheduling remain environmental assumptions.

**Audit note (changed 2026-09-10).** The former block-scope packaging axioms
`axiom_variant_conserves_records`, `axiom_variant_unique_writer`, and
`axiom_variant_schedule_terminates` were removed. As of 2026-09-11, work is
derived from routing permutation, race freedom from the explicit atomic
primitive model, and deadlock freedom from concrete proved schedules.

| Corollary | Where | Lemma |
|-----------|-------|-------|
| Same execution multiplicity for every work item | `composition_theorem.rs` §2.1 | `corollary_work_variant_invariant_from_components` |

### Routing / partitioning invariants (per-component)

| Claim | Where | Lemma |
|-------|-------|-------|
| Shard-and-gather roundtrip | Ex01, Ex02, Ex03, Ex04 | `c1_shard_gather_roundtrip`, `m2_w{0,1}_gather_roundtrip`, `q2_{q,k,v}_gather_roundtrip`, `g3_kv_gather_roundtrip` |
| Sharding disjointness | Ex01, Ex06_ep_pure, Ex06_ep/lean | `c2_sharding_disjoint`, `ep1_expert_sharding_disjoint`, `l2_local_mask_disjoint` |
| Sharding coverage | Ex06_ep_pure, Ex06_ep/lean | `ep2_expert_coverage`, `l2_local_mask_covers` |
| Offset monotonicity | Ex05 | `rt2_offset_monotonicity`, `lemma_cumsum_monotone` |
| Token conservation via offsets | Ex05, Ex09 | `rt1_token_conservation`, **`rt1_token_conservation_from_assignment`** (unconditional, via `lemma_bincount_sums_to_len`), `f1_offsets_cover_range` |
| Routing partition correctness (block size = true count; catches v8/v10) | Ex05 | `rt5_block_size_matches_count`, `rt5_wrong_size_implies_wrong_offsets` (+ `lemma_cumsum_increment`) |
| Permutation bijection over flattened routing records | Ex05 | `rt4_perm_surjective`, `rt4_perm_injective`, `rt4_each_token_exactly_once` (modulo `axiom_argsort_is_inverse_pair`; does not imply one writer per output token) |
| Top-k weight normalization | Ex05 | `rt3_topk_weight_normalization` |
| KV replication invariant | Ex04 | `g2_kv_replication_invariant` |
| Weight-loader postcondition (each variant) | Ex01, Ex02, Ex03, Ex04 | `c3_weight_loader_postcondition`, `m1_merged_weight_loader_postcondition`, `q1_qkv_weight_loader_postcondition`, `g4_gqa_weight_loader_postcondition` |
| Dispatch symmetry | Ex06_ep_pure | `ep3_dispatch_symmetric`, `ep4_token_conservation_pairwise` |
| Empty-expert handling | Ex09 | `f3_empty_expert_no_rows` |
| Replication postcondition of all_reduce | Ex06_ep/lean, `verus/axiom_base.rs` | `l5_output_replicated`, `axiom_all_reduce_sum_replicated` |

## 5. Trust boundary — every `#[verifier::external_body]` in the artifact

These are the axioms the whole paper's correctness rests on. Jun's
audit must confirm each faithfully mirrors either a Python behavior or
a well-known mathematical fact.

### Mathematical axioms (well-known facts)

| Axiom | Where | What it says |
|-------|-------|--------------|
| AXIOM_M1 | Ex01 col, Ex02-03 | matmul splits over out-dim of weight |
| AXIOM_M2 | Ex01 row, Ex02-03 | matmul splits over in-dim with sum; Ex02/Ex03 contracts explicitly require the full input and weight to reconstruct from the two shards |
| AXIOM_S1 | Ex02 | silu × mul commutes with dim-1 concat |
| AXIOM_ATTN_HEAD_LOCAL | Ex03 | attention commutes with head-shard |
| AXIOM_RI1 (length + content) | Ex04 | repeat_interleave semantics |
| AXIOM_SOFTMAX_SUM | Ex05 | softmax sums to 1 |

### Python-correspondence axioms (Verus-Python bridge)

| Axiom | Where | What it says |
|-------|-------|--------------|
| `axiom_broadcast_produces_replicated` | `axiom_base.rs` | dist.broadcast establishes Replicated |
| `axiom_all_reduce_sum_replicated` | `axiom_base.rs`, Ex06_ep/lean | dist.all_reduce(SUM) produces Replicated output |
| `axiom_all_reduce_produces_replicated` | Ex01 row | dist.all_reduce(SUM) on tp_group produces Replicated output — the Python↔Verus bridge for RowParallelLinear |
| `axiom_all_to_all_count_symmetric` | Ex06_ep_pure | send/recv counts negotiated by pre-dispatch all_to_all_single are symmetric |
| `axiom_argsort_is_inverse_pair` | `moe_baseline.rs` (RT4) | argsort returns a permutation of [0,M) with a computable inverse (library-sort property) |

There are no new `external_body` axioms in the Tier-3 bridge files.
`composition_theorem.rs` now imports the relevant Tier-1 modules, so their
documented library/collective axioms are part of its verification dependency
closure. High-level refinement/swap propositions are not theorem premises.
The remaining model-to-source boundaries are:

| Boundary | Formal role | Audit evidence required |
|----------|-------------|-------------------------|
| GQA output identifier | identifies `gqa_output_id` with the tensor returned by the source output-projection all-reduce | audit the output projection call and tensor identity; Ex04/Ex01 prove equality across participating ranks, while the theorem no longer assumes equality to an abstract `attention_forward` value |
| Residual-add and RMSNorm functions | deterministically map the replicated GQA output and common residual to the rank-indexed MoE input view | audit that all ranks apply the same source operations and parameters; numerical GQA/reference equivalence is not claimed |
| Ex05 argsort contract | produces the concrete routed order | `axiom_argsort_is_inverse_pair`; the bridge proves this implies `RoutingConsistent` |
| Ex09 source-to-DSL execution + reference representation | connects the concrete Triton wrapper to the proved DSL execution, then identifies the Python mathematical reference with shared `expert_apply` | audit dispatch construction, kernel indexing/masking, wrapper call sequence, and vector-to-scalar layout; K2/K3/F4 arithmetic is machine-checked |
| `expert_apply` | common exact expert semantic primitive | Python and Triton operations implement this abstraction |
| `scatter_primitive = AtomicIndexAdd` | models conflicting output accumulation as atomic | inspect the actual `index_add_`/kernel implementation |
| all-to-all record-transfer semantics | moves each record to its requested destination and, with swapped split sizes, returns it to its source | audit `all_to_all_variable` and the split-size construction against the global record model |
| concrete `Schedule` values | deadlock transition system's source trace | compare group, operation, call site, and order to Python |

### Per-component stubs (deferred proofs)

These are `external_body` in per-component files because their full
mechanization is future work. Each is scoped to a specific claim whose
proof structure is documented inline.

| Stub | Where | What's deferred |
|------|-------|-----------------|
| `c4_forward_correctness_general` | Ex01 col | Parameterized-over-tp_size version of C4 |
| `r4_forward_correctness_general_stub` | Ex01 row | Parameterized-over-tp_size version of R4 |
| `r2_replica_siblings_identical_kv_stub` | Ex04 | Replica-sibling attention equality |
| `r3_block_correctness_stub` | Ex04 | Full GQA-TP block correctness |
| `ep6_deadlock_free_stub` | Ex06_ep_pure | Deadlock-freedom (structural) |
| `l3_partial_output_zero_outside` | Ex06_ep/lean | Zero-outside-contributing (index_add_ postcondition) |
| `h1_attn_output_tp_replicated_stub` | Ex07 | Inherited from Ex04 R4 |
| `h3_ep_dispatch_symmetric_stub` | Ex07 | Inherited from Ex06 EP3 |
| `h4_moe_out_tp_replicated_stub` | Ex07 | All-gather postcondition |
| `h5_subgroup_deadlock_free_stub` | Ex07 | Sub-group deadlock-freedom (structural) |

**Note on stubs:** the former E1/E2/F4/L4/L6/H6, EP5/EP7, and local
composition stubs were removed. Their shared exact proofs now live in
`kernel_refinement.rs` and `schedule_refinement.rs`. Other component-local
stubs remain outside the Tier-3 derivation and must not be cited as proved
properties.

## 6. Audit priority order (Jun should read in this sequence)

For maximum paper-impact-per-hour of audit time:

### Priority 1 — shared exact execution model (45 min)

Read `verus/composition_core.rs`. Confirm the canonical flat work index really
represents `(token, top-k slot)`, `RoutingConsistent` means exact permutation,
the commutative-fold proof is valid for mathematical integers, and
`fused_rows_correct` is only a row-level kernel boundary.

### Priority 2 — refinement chain and source correspondence (2 hours)

Read `kernel_refinement.rs`, `schedule_refinement.rs`,
`component_integration.rs`, and `composition_theorem.rs`. Check that the
public block theorem calls Ex04/Ex05/Ex06/Ex09 bridges before E1/E2/F4 and
L6/H6, and that no final refinement/swap statement appears in a `requires`
clause. Then audit the remaining low-level boundaries against Python: the
GQA output identity plus residual/RMSNorm mapping, Ex05 argsort contract, Ex09 source-to-DSL and
reference-representation relations, and the mapping from schedule choice to
execution order.

### Priority 3 — the anecdote-driven proof (30 min)

Read
[`bootcamp/ex02_mlp_tp/verification/anecdote.txt`](../bootcamp/ex02_mlp_tp/verification/anecdote.txt)
alongside
[`bootcamp/ex02_mlp_tp/verification/mlp_tp.rs`](../bootcamp/ex02_mlp_tp/verification/mlp_tp.rs).
This is the concrete example the paper's Contribution 3 uses. Confirm
the anecdote accurately describes the two audit cycles.

### Priority 4 — safety goals (1 hour)

Read `verus/composition_theorem.rs` §2 and `verus/deadlock_free.rs`.

- Work conservation: audit the proof from `RoutingConsistent` multiset equality
  through canonical-work uniqueness to exact per-item multiplicity. A
  total-count theorem alone would still be insufficient.
- Deadlock freedom: compare the two-phase all-reduce and four-phase all-to-all traces to
  the actual Python collective call sites. Check group membership, operation,
  call-site identity, and ordering. The proof establishes absence of a stuck
  reachable frontier in the abstract machine.
- Data-race freedom: confirm that each modeled active output accumulation is
  implemented by atomic `index_add_`. Multiple top-k contributions may target
  one output cell; this is intentional and is no longer modeled as unique
  writing.

### Priority 5 — per-component proofs (4-6 hours)

For each `bootcamp/exNN_*/verification/*.rs`, read the `PROPERTIES.md`
first, then the `.rs`. Audit questions per file:
1. Do the properties in `PROPERTIES.md` correspond to the exercise's
   actual semantics? Read `solution.py` (or `reference_lean.py` for
   Ex06_ep) alongside.
2. Does the `.rs` file's stated theorems match `PROPERTIES.md`?
3. Are the axioms sound?
4. Are the theorem bodies free of `admit()`?

Ex01-Ex05 are structurally similar (same TP + MoE patterns). Ex06 and
Ex07 introduce distributed collectives. Ex09 has the fused-kernel
algorithmic contract.

## 7. What the paper cites, precisely

The paper's §Contribution 2 section refers to these lemmas by name:

- Meta-theorem: `theorem_shared_spec_implies_equiv` (`composition_core.rs`, re-exported by `composition_theorem.rs`)
- Schedule swap: `theorem_all_to_all_equiv_all_reduce_dp1` (`schedule_refinement.rs`, re-exported by `lean_equiv_hybrid_dp1.rs`)
- Kernel swap: `theorem_permuted_equiv_fused` (`kernel_refinement.rs`, re-exported by `naive_equiv_fused_moe.rs`)
- Component-connected 2×2 block grid: `corollary_block_variants_from_components` (`composition_theorem.rs`)
- Work conservation (Completeness + Disjointness): `theorem_token_conservation_from_components` (`composition_theorem.rs`)
- Deadlock freedom: `theorem_{all_reduce_dp1,all_to_all_dp1}_execution_deadlock_free` (`deadlock_free.rs`) and the schedule-selecting `theorem_deadlock_free` (`composition_theorem.rs`)
- Data-race freedom (via atomic conflicting writes): `theorem_data_race_free` (composition_theorem.rs)

The paper's §Methodology section refers to these files:
- Per-component proof template: `bootcamp/ex01_linear_tp/verification/`
- Audit-cycle anecdote: `bootcamp/ex02_mlp_tp/verification/anecdote.txt`
- Shared framework: `verus/axiom_base.rs` (legacy components) and `verus/{composition_core,kernel_refinement,schedule_refinement,component_integration}.rs` (Tier 3)
- Composition-theorem home: `verus/composition_theorem.rs`

## 8. Known limitations Jun should be aware of

1. **Tier 3 uses exact equality.**
   `composition_core.rs::semantic_eq` is full equality of an abstract tensor.
2. **Tier 3 proves the shared abstract execution model, not Python source
   semantics.** E1/E2/F4/L6/H6 and the block theorem are mechanically linked.
   `RoutingConsistent`, `ExpertPartitioned`, `ReplicatedInput`, and
   `fused_rows_correct` are now derived by `component_integration.rs`; the
   lower-level argsort/collective/kernel contracts, tensor identity, atomic
   primitive choice, and schedule-to-Python correspondence remain auditable
   boundaries.
3. **The Triton kernel's exact DSL semantics is modeled and verified through
   full fused F4 (`fused_kernel_dsl.rs`, K1/K2/K3/F4), but source-to-DSL
   correspondence, PTX/SASS compilation, floating-point effects, and A100
   hardware execution remain trusted boundaries.** Runtime correspondence is
   tested by `bootcamp/tests/test_ex09_fused_moe.py`; the tests are evidence,
   not part of the machine proof.
4. **Every per-component `c4_forward_correctness_general` / `r4_forward_correctness_general` is stubbed.**
   The parameterized-over-tp_size versions. Concrete tp_size=2 versions
   are proved.

The paper's threats-to-validity section lists these limitations
explicitly.

## 9. Running everything

```sh
# Set up.
export PATH=/gpfs/users/weiz/verus/verus-x86-linux:$PATH

# Verify per-component proofs.
cd /gpfs/users/weiz/jun_yang/workspace/qwen-moe
for f in bootcamp/ex01_linear_tp/verification/column_parallel.rs \
         bootcamp/ex01_linear_tp/verification/row_parallel.rs \
         bootcamp/ex02_mlp_tp/verification/mlp_tp.rs \
         bootcamp/ex03_mha_tp/verification/mha_tp.rs \
         bootcamp/ex04_gqa_tp/verification/gqa_tp.rs \
         bootcamp/ex05_moe_baseline/verification/moe_baseline.rs \
         bootcamp/ex06_ep/verification/lean.rs \
         bootcamp/ex06_ep_pure/verification/ep_pure.rs \
         bootcamp/ex07_tp_ep_hybrid/verification/hybrid.rs \
         bootcamp/ex09_fused_moe/verification/fused_moe.rs \
         bootcamp/ex09_fused_moe/verification/fused_kernel_dsl.rs; do
    dir=$(dirname "$f"); file=$(basename "$f")
    (cd "$dir" && verus "$file" 2>&1 | grep "verification results")
done

# Verify Tier 2 + Tier 3.
verus --crate-type=lib verus/axiom_base.rs
verus --crate-type=lib verus/composition_core.rs
verus --crate-type=lib verus/deadlock_free.rs
verus --crate-type=lib verus/lean_equiv_hybrid_dp1.rs
verus --crate-type=lib verus/naive_equiv_fused_moe.rs
verus --crate-type=lib verus/composition_theorem.rs
```

Expected top-level outputs are 4 (`axiom_base`), 16 (`composition_core`),
12 (`deadlock_free`), 40 (schedule crate), 22 (kernel crate), and 184
(`composition_theorem`), all with 0 errors. These numbers overlap because
imported modules are re-verified; use the 220 unique count above for the
artifact total.

Total wall-clock time: ~2 minutes on the shared pod.

## 10. Audit sign-off form

For each file Jun audits, please record in a text log:

```
File: bootcamp/exNN_XXX/verification/YYY.rs (or verus/YYY.rs)
Date audited:
Verified count matches this doc: [ Y / N ]
Axioms sound: [ Y / N — if N, list issues ]
Contracts non-decorative: [ Y / N — if N, list which ]
Proof body honest (no admit()): [ Y / N ]
Correspondence to Python solid: [ Y / N — if N, list issues ]

Comments:
```

The audit log itself is the paper's methodology evidence. Please keep
it in `verus/audit_log.md` (or a filename Jun prefers) — the paper's
Contribution 3 section will cite this log as the concrete "audit
trail" that turns the AI-drafted / human-audited workflow claim into
verifiable practice.
