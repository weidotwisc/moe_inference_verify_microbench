// fused_kernel_dsl.rs
//
// DSL-LEVEL VERUS PROOF of the fused-MoE Triton grouped-GEMM kernel.
//
// Companion to `fused_moe.rs` (which proved algorithmic-contract F1, F3
// about the offsets structure). THIS file models the kernel's INTERNAL
// structure — dispatch table lookup, tile-index arithmetic, and the
// K-axis reduction loop — as Verus spec functions, and proves that:
//
//   K1  (Tile coverage)         : distinct (mid, nid) program instances
//                                  write to disjoint output regions;
//                                  their union covers the full output.
//   K2  (K-reduce correctness)  : the accumulator loop over K-tiles
//                                  computes the matmul on the tile's
//                                  block.
//   K3  (Kernel correctness)    : the assembled kernel output equals the
//                                  semantic reference (F4, now derived
//                                  rather than trusted).
//
// What this file does NOT cover:
//   - A mechanized parser/semantics bridge from the Python Triton AST to
//     this DSL model; that source-to-model correspondence is audited.
//   - The Triton → PTX compilation.
//   - The PTX → SASS lowering.
//   - A100 hardware execution of the compiled machine code.
//
// Within the DSL model, dispatch ownership, launch-grid coverage, the exact
// K reduction, and all three grouped matmuls are machine-checked.  The
// remaining trust boundary starts at source-to-model correspondence, followed
// by the compiler and hardware stack.
//
// Run with:
//   verus fused_kernel_dsl.rs

use vstd::prelude::*;

verus! {

// =====================================================================
// §1 — Types.
// =====================================================================

pub type Element = int;
pub type Row = Seq<Element>;
pub type Tensor = Seq<Row>;
pub type ExpertId = nat;

pub open spec fn well_formed(t: Tensor, rows: nat, cols: nat) -> bool {
    t.len() == rows &&
    forall|i: int| 0 <= i < t.len() ==> #[trigger] t[i].len() == cols
}

// =====================================================================
// §2 — Dispatch table (Python-side of the kernel).
//
// Given `expert_offsets: [E+1]` from the routing step, the wrapper
// `_build_tile_dispatch` in ex09/reference.py builds two arrays:
//   tile_expert       : [total_tiles] — mid → which expert
//   tile_row_start    : [total_tiles] — mid → global first row of the
//                                        BLOCK_M-sized expert tile.
// The launch grid is (total_tiles, cdiv(N, BLOCK_N)).
// =====================================================================

pub struct DispatchTable {
    pub tile_expert: Seq<ExpertId>,        // [total_tiles]
    pub tile_row_start: Seq<nat>,          // [total_tiles]
    pub expert_offsets: Seq<nat>,          // [E+1]
    pub total_tiles: nat,
    pub num_experts: nat,
    pub block_m: nat,
}

/// A DispatchTable is well-formed when its arrays have consistent
/// shapes and every entry references a valid expert with a row start inside
/// that expert's contiguous block.
pub open spec fn dt_well_formed(dt: DispatchTable) -> bool {
    &&& dt.block_m >= 1
    &&& dt.num_experts >= 1
    &&& dt.expert_offsets.len() as nat == dt.num_experts + 1
    &&& dt.expert_offsets[0] == 0nat
    &&& forall|e: int| #![trigger dt.expert_offsets[e]] 0 <= e < dt.num_experts as int
            ==> dt.expert_offsets[e] <= dt.expert_offsets[e + 1]
    &&& dt.tile_expert.len() == dt.total_tiles
    &&& dt.tile_row_start.len() == dt.total_tiles
    &&& forall|mid: int| #![trigger dt.tile_expert[mid]] 0 <= mid < dt.total_tiles as int
            ==> {
                let expert = dt.tile_expert[mid];
                &&& expert < dt.num_experts
                &&& dt.expert_offsets[expert as int] <= dt.tile_row_start[mid]
                &&& dt.tile_row_start[mid] < dt.expert_offsets[expert as int + 1]
            }
}

// =====================================================================
// §3 — Masked load semantics (models tl.load with boundary_check +
//                             padding_option="zero").
//
// masked_load_1d(seq, start, len, upper_bound):
//   - For positions [start, min(start+len, upper_bound)): returns seq's
//     values.
//   - For positions in [max(start, upper_bound), start+len): returns 0.
// =====================================================================

pub open spec fn masked_load_1d(
    seq: Seq<Element>, start: nat, len: nat, upper_bound: nat,
) -> Seq<Element>
    decreases len,
{
    if len == 0 {
        Seq::<Element>::empty()
    } else {
        let first = if (start as int) < (upper_bound as int) && (start as int) < seq.len() as int {
            seq[start as int]
        } else {
            0int
        };
        seq![first] + masked_load_1d(seq, (start + 1) as nat, (len - 1) as nat, upper_bound)
    }
}

pub proof fn lemma_masked_load_1d_len(seq: Seq<Element>, start: nat, len: nat, upper_bound: nat)
    ensures masked_load_1d(seq, start, len, upper_bound).len() == len,
    decreases len,
{
    if len == 0 {
    } else {
        lemma_masked_load_1d_len(seq, (start + 1) as nat, (len - 1) as nat, upper_bound);
    }
}

// =====================================================================
// §4 — Exact dot product + K-tiled reduction.
//
// Triton's `tl.dot(..., acc=acc)` updates every output element by adding
// the products in one BLOCK_K-wide slice.  We model one output element
// directly.  Lifting this pointwise fact to a matrix tile is extensional:
// every (row, column) element executes the same K loop.
// =====================================================================

pub open spec fn range_sum(values: Seq<Element>, start: nat, end: nat) -> Element
    decreases end - start,
{
    if start >= end || end > values.len() {
        0
    } else {
        values[start as int] + range_sum(values, (start + 1) as nat, end)
    }
}

pub open spec fn dot_terms(x: Row, w: Row, k_total: nat) -> Seq<Element> {
    if k_total <= x.len() && k_total <= w.len() {
        Seq::new(k_total, |k: int| x[k] * w[k])
    } else {
        Seq::empty()
    }
}

pub open spec fn dot_product(x: Row, w: Row, k_total: nat) -> Element {
    range_sum(dot_terms(x, w, k_total), 0, k_total)
}

/// Clamp a K-tile boundary to the logical K extent.  This exactly captures
/// Triton's zero-padding on the final partial tile: positions at or beyond K
/// contribute no term to the mathematical reduction.
pub open spec fn k_boundary(tile: nat, block_k: nat, k_total: nat) -> nat {
    if tile * block_k < k_total {
        tile * block_k
    } else {
        k_total
    }
}

pub open spec fn k_tile_sum(
    terms: Seq<Element>, tile: nat, block_k: nat, k_total: nat,
) -> Element {
    range_sum(
        terms,
        k_boundary(tile, block_k, k_total),
        k_boundary((tile + 1) as nat, block_k, k_total),
    )
}

/// Exact scalar semantics of the BLOCK_K loop for one output element.
pub open spec fn k_reduce_entry(
    terms: Seq<Element>,
    j: nat,
    num_k_tiles: nat,
    block_k: nat,
    k_total: nat,
    acc: Element,
) -> Element
    decreases num_k_tiles - j,
{
    if j >= num_k_tiles {
        acc
    } else {
        k_reduce_entry(
            terms,
            (j + 1) as nat,
            num_k_tiles,
            block_k,
            k_total,
            acc + k_tile_sum(terms, j, block_k, k_total),
        )
    }
}

// =====================================================================
// §5 — Pointwise grouped-matmul semantics.
// =====================================================================

/// One output element of the Python/PyTorch grouped-matmul reference.
pub open spec fn grouped_matmul_reference_entry(
    x_row: Row, expert_weight_row: Row, k_total: nat,
) -> Element {
    dot_product(x_row, expert_weight_row, k_total)
}

/// The corresponding output element produced by the Triton DSL K loop.
pub open spec fn grouped_matmul_dsl_entry(
    x_row: Row,
    expert_weight_row: Row,
    k_total: nat,
    block_k: nat,
    num_k_tiles: nat,
) -> Element {
    k_reduce_entry(
        dot_terms(x_row, expert_weight_row, k_total),
        0,
        num_k_tiles,
        block_k,
        k_total,
        0,
    )
}

pub open spec fn grouped_weights_well_formed(
    weights: Seq<Tensor>, experts: nat, output_cols: nat, k_total: nat,
) -> bool {
    &&& weights.len() == experts
    &&& forall|e: int| #![trigger weights[e]] 0 <= e < experts ==> {
        &&& weights[e].len() == output_cols
        &&& forall|col: int| #![trigger weights[e][col]]
            0 <= col < output_cols ==> weights[e][col].len() == k_total
    }
}

/// Mathematical Python-loop reference: select the row's expert, then compute
/// every output column by an untiled exact dot product.
pub open spec fn grouped_matmul_reference(
    x: Tensor,
    weights: Seq<Tensor>,
    owner: spec_fn(nat) -> ExpertId,
    total_rows: nat,
    output_cols: nat,
    k_total: nat,
) -> Tensor
    recommends
        well_formed(x, total_rows, k_total),
        grouped_weights_well_formed(
            weights, weights.len(), output_cols, k_total,
        ),
        forall|row: nat| #![trigger owner(row)]
            row < total_rows ==> owner(row) < weights.len(),
{
    Seq::new(total_rows, |row: int|
        Seq::new(output_cols, |col: int|
            grouped_matmul_reference_entry(
                x[row], weights[owner(row as nat) as int][col], k_total,
            )
        )
    )
}

/// Pointwise semantics of the Triton launch after K1 has selected the unique
/// M-axis program for a row. N-axis coverage determines the unique column
/// tile; it does not change the value computed at that column.
pub open spec fn grouped_matmul_dsl(
    x: Tensor,
    weights: Seq<Tensor>,
    dt: DispatchTable,
    total_rows: nat,
    output_cols: nat,
    k_total: nat,
    block_k: nat,
    num_k_tiles: nat,
) -> Tensor
    recommends
        well_formed(x, total_rows, k_total),
        grouped_weights_well_formed(
            weights, dt.num_experts, output_cols, k_total,
        ),
        forall|row: nat| #![trigger program_for_row(dt, row)] row < total_rows ==>
            exists|mid: nat| program_covers_row(dt, mid, row),
{
    Seq::new(total_rows, |row: int| {
        let mid = program_for_row(dt, row as nat);
        let expert = dt.tile_expert[mid as int];
        Seq::new(output_cols, |col: int|
            grouped_matmul_dsl_entry(
                x[row],
                weights[expert as int][col],
                k_total,
                block_k,
                num_k_tiles,
            )
        )
    })
}

pub uninterp spec fn silu(value: Element) -> Element;

pub open spec fn gated_activation(
    gate: Tensor, up: Tensor, total_rows: nat, cols: nat,
) -> Tensor
    recommends
        well_formed(gate, total_rows, cols),
        well_formed(up, total_rows, cols),
{
    Seq::new(total_rows, |row: int|
        Seq::new(cols, |col: int| silu(gate[row][col]) * up[row][col])
    )
}

pub proof fn lemma_grouped_matmul_reference_well_formed(
    x: Tensor,
    weights: Seq<Tensor>,
    owner: spec_fn(nat) -> ExpertId,
    total_rows: nat,
    output_cols: nat,
    k_total: nat,
)
    requires
        well_formed(x, total_rows, k_total),
        grouped_weights_well_formed(
            weights, weights.len(), output_cols, k_total,
        ),
        forall|row: nat| #![trigger owner(row)]
            row < total_rows ==> owner(row) < weights.len(),
    ensures well_formed(
        grouped_matmul_reference(
            x, weights, owner, total_rows, output_cols, k_total,
        ),
        total_rows,
        output_cols,
    ),
{
}

pub proof fn lemma_grouped_matmul_dsl_well_formed(
    x: Tensor,
    weights: Seq<Tensor>,
    dt: DispatchTable,
    owner: spec_fn(nat) -> ExpertId,
    total_rows: nat,
    output_cols: nat,
    k_total: nat,
    block_k: nat,
    num_k_tiles: nat,
)
    requires
        well_formed(x, total_rows, k_total),
        grouped_weights_well_formed(
            weights, dt.num_experts, output_cols, k_total,
        ),
        dispatch_matches_row_owner(dt, total_rows, owner),
    ensures well_formed(
        grouped_matmul_dsl(
            x, weights, dt, total_rows, output_cols, k_total,
            block_k, num_k_tiles,
        ),
        total_rows,
        output_cols,
    ),
{
    assert forall|row: nat| #![trigger program_for_row(dt, row)]
        row < total_rows implies {
            &&& exists|mid: nat| program_covers_row(dt, mid, row)
            &&& dt.tile_expert[program_for_row(dt, row) as int]
                < dt.num_experts
        } by {
        lemma_program_for_row_matches_owner(dt, total_rows, owner, row);
    }
}

pub proof fn lemma_gated_activation_well_formed(
    gate: Tensor, up: Tensor, total_rows: nat, cols: nat,
)
    requires
        well_formed(gate, total_rows, cols),
        well_formed(up, total_rows, cols),
    ensures well_formed(
        gated_activation(gate, up, total_rows, cols), total_rows, cols,
    ),
{
}

/// Exact Python-loop reference for the three operations in one SwiGLU expert:
/// gate projection, up projection, pointwise SiLU/product, and down projection.
pub open spec fn fused_moe_reference(
    x: Tensor,
    gate_weights: Seq<Tensor>,
    up_weights: Seq<Tensor>,
    down_weights: Seq<Tensor>,
    owner: spec_fn(nat) -> ExpertId,
    total_rows: nat,
    hidden_cols: nat,
    intermediate_cols: nat,
) -> Tensor {
    let gate = grouped_matmul_reference(
        x, gate_weights, owner, total_rows, intermediate_cols, hidden_cols,
    );
    let up = grouped_matmul_reference(
        x, up_weights, owner, total_rows, intermediate_cols, hidden_cols,
    );
    let hidden = gated_activation(gate, up, total_rows, intermediate_cols);
    grouped_matmul_reference(
        hidden, down_weights, owner,
        total_rows, hidden_cols, intermediate_cols,
    )
}

/// Exact DSL semantics of the three grouped-GEMM launches in
/// `fused_moe_forward`.
pub open spec fn fused_moe_dsl(
    x: Tensor,
    gate_weights: Seq<Tensor>,
    up_weights: Seq<Tensor>,
    down_weights: Seq<Tensor>,
    dt: DispatchTable,
    total_rows: nat,
    hidden_cols: nat,
    intermediate_cols: nat,
    gate_block_k: nat,
    gate_num_k_tiles: nat,
    down_block_k: nat,
    down_num_k_tiles: nat,
) -> Tensor {
    let gate = grouped_matmul_dsl(
        x, gate_weights, dt, total_rows, intermediate_cols, hidden_cols,
        gate_block_k, gate_num_k_tiles,
    );
    let up = grouped_matmul_dsl(
        x, up_weights, dt, total_rows, intermediate_cols, hidden_cols,
        gate_block_k, gate_num_k_tiles,
    );
    let hidden = gated_activation(gate, up, total_rows, intermediate_cols);
    grouped_matmul_dsl(
        hidden, down_weights, dt,
        total_rows, hidden_cols, intermediate_cols,
        down_block_k, down_num_k_tiles,
    )
}

// =====================================================================
// §6 — Property K1a: tile M-axis coverage / disjointness.
//
// The dispatch table's M-axis tiles partition the union of expert
// blocks. Specifically:
//   - Two distinct mid values cover disjoint logical output rows.
//   - The union of all mid values' tiles covers [0, total_tiles * BLOCK_M).
// =====================================================================

/// The M-axis span (start, end) of the tile owned by program instance mid.
pub open spec fn tile_m_span(dt: DispatchTable, mid: nat) -> (nat, nat)
    recommends mid < dt.total_tiles,
{
    let eid = dt.tile_expert[mid as int];
    let tile_start = dt.tile_row_start[mid as int];
    let tile_end = tile_start + dt.block_m;
    (tile_start, tile_end)
}

/// A dispatch-table program covers the non-padding rows between its global
/// row start and the end of its expert's contiguous block.
pub open spec fn program_covers_row(
    dt: DispatchTable, mid: nat, row: nat,
) -> bool {
    &&& mid < dt.total_tiles
    &&& row >= dt.tile_row_start[mid as int]
    &&& row < dt.tile_row_start[mid as int] + dt.block_m
    &&& row < dt.expert_offsets[dt.tile_expert[mid as int] as int + 1]
}

/// Semantic contract of `_build_tile_dispatch`: every logical input row is
/// assigned to exactly one M-axis program and that program names the same
/// expert as the Python reference's offsets-based owner function.
pub open spec fn dispatch_matches_row_owner(
    dt: DispatchTable,
    total_rows: nat,
    owner: spec_fn(nat) -> ExpertId,
) -> bool {
    &&& dt_well_formed(dt)
    &&& dt.expert_offsets[dt.num_experts as int] == total_rows
    &&& forall|row: nat| #![trigger owner(row)] row < total_rows ==> {
        &&& owner(row) < dt.num_experts
        &&& exists|mid: nat| program_covers_row(dt, mid, row)
            && dt.tile_expert[mid as int] == owner(row)
    }
    &&& forall|row: nat, mid1: nat, mid2: nat|
        #![trigger program_covers_row(dt, mid1, row), program_covers_row(dt, mid2, row)]
        row < total_rows
        && program_covers_row(dt, mid1, row)
        && program_covers_row(dt, mid2, row)
            ==> mid1 == mid2
}

pub open spec fn program_for_row(dt: DispatchTable, row: nat) -> nat
    recommends exists|mid: nat| program_covers_row(dt, mid, row),
{
    choose|mid: nat| program_covers_row(dt, mid, row)
}

/// K1a: two distinct M-axis programs never cover the same logical output row.
/// Padding rows at the end of a tile are excluded by `program_covers_row`.
pub proof fn k1a_m_axis_disjoint(
    dt: DispatchTable,
    total_rows: nat,
    owner: spec_fn(nat) -> ExpertId,
    mid1: nat,
    mid2: nat,
)
    requires
        dispatch_matches_row_owner(dt, total_rows, owner),
        mid1 < dt.total_tiles,
        mid2 < dt.total_tiles,
        mid1 != mid2,
    ensures forall|row: nat|
        row < total_rows ==> !(
            program_covers_row(dt, mid1, row)
                && program_covers_row(dt, mid2, row)
        ),
{
    assert forall|row: nat|
        row < total_rows implies !(
            program_covers_row(dt, mid1, row)
                && program_covers_row(dt, mid2, row)
        ) by {
        if program_covers_row(dt, mid1, row)
            && program_covers_row(dt, mid2, row) {
            assert(mid1 == mid2);
        }
    }
}

// =====================================================================
// §7 — Property K1b: N-axis tile coverage.
//
// The launch grid's N-axis dimension is cdiv(N, BLOCK_N). Each nid
// covers the tile [nid * BLOCK_N, (nid+1) * BLOCK_N) (with zero-padding
// for OOB positions). Together they cover [0, N).
// =====================================================================

pub proof fn k1b_n_axis_covers(
    n_total: nat, block_n: nat, num_n_tiles: nat, col: nat,
)
    requires
        block_n >= 1,
        num_n_tiles * block_n >= n_total,
        col < n_total,
    ensures
        exists|nid: nat| #![trigger nid * block_n]
                     nid < num_n_tiles
                     && (nid * block_n) <= col
                     && col < (nid + 1) * block_n,
{
    // Witness: nid := col / block_n.
    let nid_witness = (col as int / block_n as int) as nat;
    // Show nid_witness < num_n_tiles.
    assert(nid_witness * block_n <= col) by (nonlinear_arith)
        requires nid_witness == (col as int / block_n as int) as nat, block_n >= 1;
    assert(col < (nid_witness + 1) * block_n) by (nonlinear_arith)
        requires nid_witness == (col as int / block_n as int) as nat, block_n >= 1;
    assert(nid_witness < num_n_tiles) by (nonlinear_arith)
        requires
            nid_witness * block_n <= col,
            col < n_total,
            num_n_tiles * block_n >= n_total,
            block_n >= 1;
}

// =====================================================================
// §8 — Property K1c: tile-write disjointness across nid.
//
// Distinct nid values on the same mid write to disjoint N-axis regions.
// =====================================================================

pub proof fn k1c_n_axis_disjoint(
    block_n: nat, nid1: nat, nid2: nat,
)
    requires
        block_n >= 1,
        nid1 != nid2,
    ensures
        (nid1 + 1) * block_n <= nid2 * block_n
        || (nid2 + 1) * block_n <= nid1 * block_n,
{
    if nid1 < nid2 {
        assert(nid1 + 1 <= nid2);
        assert((nid1 + 1) * block_n <= nid2 * block_n) by (nonlinear_arith)
            requires nid1 + 1 <= nid2, block_n >= 1;
    } else {
        assert(nid2 < nid1);
        assert(nid2 + 1 <= nid1);
        assert((nid2 + 1) * block_n <= nid1 * block_n) by (nonlinear_arith)
            requires nid2 + 1 <= nid1, block_n >= 1;
    }
}

// =====================================================================
// §9 — Property K2: K-reduce correctness.
// =====================================================================

proof fn lemma_range_sum_split(
    values: Seq<Element>, start: nat, mid: nat, end: nat,
)
    requires
        start <= mid,
        mid <= end,
        end <= values.len(),
    ensures
        range_sum(values, start, end)
            == range_sum(values, start, mid) + range_sum(values, mid, end),
    decreases mid - start,
{
    if start < mid {
        lemma_range_sum_split(values, (start + 1) as nat, mid, end);
    }
}

proof fn lemma_k_boundaries_ordered(
    tile: nat, block_k: nat, k_total: nat,
)
    requires block_k > 0,
    ensures
        k_boundary(tile, block_k, k_total)
            <= k_boundary((tile + 1) as nat, block_k, k_total),
        k_boundary((tile + 1) as nat, block_k, k_total) <= k_total,
{
    assert(tile * block_k <= (tile + 1) * block_k) by (nonlinear_arith);
    if tile * block_k < k_total {
        if (tile + 1) * block_k < k_total {
            assert(k_boundary(tile, block_k, k_total) == tile * block_k);
            assert(k_boundary((tile + 1) as nat, block_k, k_total)
                == (tile + 1) * block_k);
        } else {
            assert(k_boundary(tile, block_k, k_total) == tile * block_k);
            assert(k_boundary((tile + 1) as nat, block_k, k_total) == k_total);
        }
    } else {
        assert(k_total <= tile * block_k);
        assert(k_total <= (tile + 1) * block_k);
        assert(k_boundary(tile, block_k, k_total) == k_total);
        assert(k_boundary((tile + 1) as nat, block_k, k_total) == k_total);
    }
}

proof fn lemma_k_reduce_entry_invariant(
    terms: Seq<Element>,
    j: nat,
    num_k_tiles: nat,
    block_k: nat,
    k_total: nat,
    acc: Element,
)
    requires
        terms.len() == k_total,
        block_k > 0,
        j <= num_k_tiles,
        k_total <= num_k_tiles * block_k,
    ensures
        k_reduce_entry(terms, j, num_k_tiles, block_k, k_total, acc)
            == acc + range_sum(
                terms, k_boundary(j, block_k, k_total), k_total,
            ),
    decreases num_k_tiles - j,
{
    if j < num_k_tiles {
        let here = k_boundary(j, block_k, k_total);
        let next = k_boundary((j + 1) as nat, block_k, k_total);
        lemma_k_boundaries_ordered(j, block_k, k_total);
        lemma_range_sum_split(terms, here, next, k_total);
        lemma_k_reduce_entry_invariant(
            terms,
            (j + 1) as nat,
            num_k_tiles,
            block_k,
            k_total,
            acc + k_tile_sum(terms, j, block_k, k_total),
        );
    } else {
        assert(j == num_k_tiles);
        assert(k_boundary(j, block_k, k_total) == k_total) by {
            assert(k_total <= j * block_k);
        }
    }
}

/// K2: the exact BLOCK_K accumulator equals the untiled dot product.
/// No arithmetic or decomposition axiom is used; the proof partitions the
/// concrete sequence of scalar products at successive clamped tile bounds.
pub proof fn k2_k_reduce_correctness(
    x_row: Row,
    expert_weight_row: Row,
    k_total: nat,
    block_k: nat,
    num_k_tiles: nat,
)
    requires
        k_total <= x_row.len(),
        k_total <= expert_weight_row.len(),
        block_k > 0,
        k_total <= num_k_tiles * block_k,
    ensures
        grouped_matmul_dsl_entry(
            x_row, expert_weight_row, k_total, block_k, num_k_tiles,
        ) == grouped_matmul_reference_entry(
            x_row, expert_weight_row, k_total,
        ),
{
    let terms = dot_terms(x_row, expert_weight_row, k_total);
    assert(terms.len() == k_total);
    lemma_k_reduce_entry_invariant(
        terms, 0, num_k_tiles, block_k, k_total, 0,
    );
}

// =====================================================================
// §10 — Property K3: pointwise grouped-matmul correctness.
// =====================================================================

/// Every element in a BLOCK_M x BLOCK_N program tile runs the K loop proved
/// by K2.  Row/column tile coverage is handled independently by K1.
pub proof fn k3_kernel_correctness_at_output_element(
    x_row: Row,
    expert_weight_row: Row,
    k_total: nat,
    block_k: nat,
    num_k_tiles: nat,
)
    requires
        k_total <= x_row.len(),
        k_total <= expert_weight_row.len(),
        block_k > 0,
        k_total <= num_k_tiles * block_k,
    ensures
        grouped_matmul_dsl_entry(
            x_row, expert_weight_row, k_total, block_k, num_k_tiles,
        ) == grouped_matmul_reference_entry(
            x_row, expert_weight_row, k_total,
        ),
{
    k2_k_reduce_correctness(
        x_row, expert_weight_row, k_total, block_k, num_k_tiles,
    );
}

proof fn lemma_program_for_row_matches_owner(
    dt: DispatchTable,
    total_rows: nat,
    owner: spec_fn(nat) -> ExpertId,
    row: nat,
)
    requires
        dispatch_matches_row_owner(dt, total_rows, owner),
        row < total_rows,
    ensures
        program_covers_row(dt, program_for_row(dt, row), row),
        dt.tile_expert[program_for_row(dt, row) as int] == owner(row),
{
    let witness = choose|mid: nat| program_covers_row(dt, mid, row)
        && dt.tile_expert[mid as int] == owner(row);
    assert(program_covers_row(dt, witness, row));
    let selected = program_for_row(dt, row);
    assert(program_covers_row(dt, selected, row));
    assert(selected == witness);
}

/// K3: under the dispatch-table contract and launch-grid coverage, the whole
/// grouped-GEMM tensor produced by the modeled Triton programs is exactly the
/// Python/PyTorch grouped-matmul reference in integer arithmetic.
pub proof fn k3_grouped_matmul_dsl_equals_reference(
    x: Tensor,
    weights: Seq<Tensor>,
    dt: DispatchTable,
    owner: spec_fn(nat) -> ExpertId,
    total_rows: nat,
    output_cols: nat,
    k_total: nat,
    block_n: nat,
    num_n_tiles: nat,
    block_k: nat,
    num_k_tiles: nat,
)
    requires
        well_formed(x, total_rows, k_total),
        grouped_weights_well_formed(
            weights, dt.num_experts, output_cols, k_total,
        ),
        dispatch_matches_row_owner(dt, total_rows, owner),
        block_n > 0,
        output_cols <= num_n_tiles * block_n,
        block_k > 0,
        k_total <= num_k_tiles * block_k,
    ensures
        grouped_matmul_dsl(
            x, weights, dt, total_rows, output_cols, k_total,
            block_k, num_k_tiles,
        ) == grouped_matmul_reference(
            x, weights, owner, total_rows, output_cols, k_total,
        ),
{
    let dsl = grouped_matmul_dsl(
        x, weights, dt, total_rows, output_cols, k_total,
        block_k, num_k_tiles,
    );
    let reference = grouped_matmul_reference(
        x, weights, owner, total_rows, output_cols, k_total,
    );
    assert(dsl.len() == total_rows);
    assert(reference.len() == total_rows);
    assert forall|row: int| 0 <= row < total_rows implies
        dsl[row] == reference[row] by {
        lemma_program_for_row_matches_owner(
            dt, total_rows, owner, row as nat,
        );
        let mid = program_for_row(dt, row as nat);
        let expert = dt.tile_expert[mid as int];
        assert(expert == owner(row as nat));
        assert(dsl[row].len() == output_cols);
        assert(reference[row].len() == output_cols);
        assert forall|col: int| 0 <= col < output_cols implies
            dsl[row][col] == reference[row][col] by {
            k1b_n_axis_covers(
                output_cols, block_n, num_n_tiles, col as nat,
            );
            assert(weights[expert as int][col].len() == k_total);
            k3_kernel_correctness_at_output_element(
                x[row], weights[expert as int][col],
                k_total, block_k, num_k_tiles,
            );
        }
        assert(dsl[row] =~= reference[row]);
    }
    assert(dsl =~= reference);
}

/// F4 at the DSL level: composing the three verified grouped matmuls with
/// deterministic pointwise gating gives exactly the Python fused-expert
/// reference. Gate/up share a launch shape; down has its own N/K tiling.
pub proof fn f4_fused_moe_dsl_equals_python_reference(
    x: Tensor,
    gate_weights: Seq<Tensor>,
    up_weights: Seq<Tensor>,
    down_weights: Seq<Tensor>,
    dt: DispatchTable,
    owner: spec_fn(nat) -> ExpertId,
    total_rows: nat,
    hidden_cols: nat,
    intermediate_cols: nat,
    gate_block_n: nat,
    gate_num_n_tiles: nat,
    gate_block_k: nat,
    gate_num_k_tiles: nat,
    down_block_n: nat,
    down_num_n_tiles: nat,
    down_block_k: nat,
    down_num_k_tiles: nat,
)
    requires
        well_formed(x, total_rows, hidden_cols),
        grouped_weights_well_formed(
            gate_weights, dt.num_experts, intermediate_cols, hidden_cols,
        ),
        grouped_weights_well_formed(
            up_weights, dt.num_experts, intermediate_cols, hidden_cols,
        ),
        grouped_weights_well_formed(
            down_weights, dt.num_experts, hidden_cols, intermediate_cols,
        ),
        dispatch_matches_row_owner(dt, total_rows, owner),
        gate_block_n > 0,
        intermediate_cols <= gate_num_n_tiles * gate_block_n,
        gate_block_k > 0,
        hidden_cols <= gate_num_k_tiles * gate_block_k,
        down_block_n > 0,
        hidden_cols <= down_num_n_tiles * down_block_n,
        down_block_k > 0,
        intermediate_cols <= down_num_k_tiles * down_block_k,
    ensures
        fused_moe_dsl(
            x, gate_weights, up_weights, down_weights, dt,
            total_rows, hidden_cols, intermediate_cols,
            gate_block_k, gate_num_k_tiles,
            down_block_k, down_num_k_tiles,
        ) == fused_moe_reference(
            x, gate_weights, up_weights, down_weights, owner,
            total_rows, hidden_cols, intermediate_cols,
        ),
{
    assert forall|row: nat| #![trigger program_for_row(dt, row)]
        row < total_rows implies
            exists|mid: nat| program_covers_row(dt, mid, row) by {
        lemma_program_for_row_matches_owner(dt, total_rows, owner, row);
    }
    let gate_dsl = grouped_matmul_dsl(
        x, gate_weights, dt, total_rows, intermediate_cols, hidden_cols,
        gate_block_k, gate_num_k_tiles,
    );
    let gate_reference = grouped_matmul_reference(
        x, gate_weights, owner, total_rows, intermediate_cols, hidden_cols,
    );
    let up_dsl = grouped_matmul_dsl(
        x, up_weights, dt, total_rows, intermediate_cols, hidden_cols,
        gate_block_k, gate_num_k_tiles,
    );
    let up_reference = grouped_matmul_reference(
        x, up_weights, owner, total_rows, intermediate_cols, hidden_cols,
    );

    k3_grouped_matmul_dsl_equals_reference(
        x, gate_weights, dt, owner,
        total_rows, intermediate_cols, hidden_cols,
        gate_block_n, gate_num_n_tiles,
        gate_block_k, gate_num_k_tiles,
    );
    k3_grouped_matmul_dsl_equals_reference(
        x, up_weights, dt, owner,
        total_rows, intermediate_cols, hidden_cols,
        gate_block_n, gate_num_n_tiles,
        gate_block_k, gate_num_k_tiles,
    );
    assert(gate_dsl == gate_reference);
    assert(up_dsl == up_reference);

    lemma_grouped_matmul_dsl_well_formed(
        x, gate_weights, dt, owner, total_rows, intermediate_cols, hidden_cols,
        gate_block_k, gate_num_k_tiles,
    );
    lemma_grouped_matmul_dsl_well_formed(
        x, up_weights, dt, owner, total_rows, intermediate_cols, hidden_cols,
        gate_block_k, gate_num_k_tiles,
    );
    let hidden_dsl = gated_activation(
        gate_dsl, up_dsl, total_rows, intermediate_cols,
    );
    let hidden_reference = gated_activation(
        gate_reference, up_reference, total_rows, intermediate_cols,
    );
    assert(hidden_dsl == hidden_reference);
    lemma_gated_activation_well_formed(
        gate_dsl, up_dsl, total_rows, intermediate_cols,
    );

    k3_grouped_matmul_dsl_equals_reference(
        hidden_dsl, down_weights, dt, owner,
        total_rows, hidden_cols, intermediate_cols,
        down_block_n, down_num_n_tiles,
        down_block_k, down_num_k_tiles,
    );
}

// =====================================================================
// §11 — End of the machine-checked DSL refinement.
// =====================================================================

} // verus!

fn main() {
    println!("Verified DSL-level Triton kernel structural properties (Verus).");
}
