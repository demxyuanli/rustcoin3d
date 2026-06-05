//! Curve and surface geometry (B-Rep evaluation kernel).

pub mod bspline;
pub mod project;
pub mod curve2d;

pub mod curve_eval;
pub mod surface_eval;

pub use curve_eval::*;
pub use surface_eval::*;
