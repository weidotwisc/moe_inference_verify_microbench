// Standalone crate root for exact all-reduce/all-to-all equivalence at DP=1.
//
// Run with:
//   verus --crate-type=lib verus/lean_equiv_hybrid_dp1.rs

#[path = "composition_core.rs"]
mod composition_core;
pub use composition_core::*;

#[path = "kernel_refinement.rs"]
mod kernel_refinement;
pub use kernel_refinement::*;

#[path = "schedule_refinement.rs"]
mod schedule_refinement;
pub use schedule_refinement::*;
