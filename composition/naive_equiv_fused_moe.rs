// Standalone crate root for the shared exact E1/E2/F4 kernel refinements.
//
// Run with:
//   verus --crate-type=lib verus/naive_equiv_fused_moe.rs

#[path = "composition_core.rs"]
mod composition_core;
pub use composition_core::*;

#[path = "kernel_refinement.rs"]
mod kernel_refinement;
pub use kernel_refinement::*;
