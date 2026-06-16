//! Curve and surface geometry (B-Rep evaluation kernel).

pub mod bspline;
pub mod project;
pub mod curve2d;
pub mod properties;

pub mod curve_eval;
pub mod surface_eval;

pub use curve_eval::*;
pub use surface_eval::*;
pub use curve2d::Curve2d;
pub use properties::{face_area, solid_volume};
