//! Concrete FaceMesher strategy implementations.
//!
//! Each strategy wraps an existing mesh algorithm behind the `FaceMesher` trait,
//! enabling the `FaceStrategyChain` to dispatch per-face with retry hints.

pub mod closed_parametric;
pub mod trimmed_cdt;
pub mod ruled_strip;
pub mod parametric_grid;

pub use closed_parametric::ClosedParametricMesher;
pub use trimmed_cdt::TrimmedCdtMesher;
pub use ruled_strip::RuledStripMesher;
pub use parametric_grid::ParametricGridMesher;
