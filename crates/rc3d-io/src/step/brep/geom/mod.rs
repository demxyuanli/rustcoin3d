//! Unified curve and surface geometry (B-Rep evaluation kernel).

pub mod bspline;
pub mod nurbs_build;

mod curve_surface;

pub use curve_surface::*;
