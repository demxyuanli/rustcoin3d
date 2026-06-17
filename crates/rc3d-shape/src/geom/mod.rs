//! # B-Rep Geometry Evaluation Kernel
//!
//! Parametric curve and surface definitions with evaluation, projection,
//! and geometric property queries.
//!
//! ## Architecture
//! - `curve_eval` — `CurveGeom` enum (Line/Circle/Ellipse/BSpline/Offset/Trimmed)
//!   with point/tangent/curvature evaluation at parameter `t`
//! - `surface_eval` — `SurfaceGeom` enum (Plane/Cylinder/Cone/Sphere/Torus/BSpline/Revolution/Extrusion)
//!   with point/normal/curvature evaluation at parameter `(u, v)`
//! - `curve2d` — `Curve2d` parametric curve in 2D (pcurves, trimming contours)
//! - `bspline` — B-spline basis functions, knot insertion, degree elevation
//! - `project` — point-to-curve and point-to-surface projection (Newton iteration)
//! - `properties` — `face_area()`, `solid_volume()` geometric property computation
//!
//! ## OCC alignment
//! Corresponds to OpenCASCADE `Geom` (3D curves), `Geom2d` (2D curves),
//! `GeomAdaptor` (unified evaluator), `GeomAPI_ProjectPointOnCurve`/`Surf`,
//! and `BRepGProp` (global properties).
//!
//! ## Usage
//! ```ignore
//! use rc3d_shape::geom::{CurveGeom, SurfaceGeom};
//! use rc3d_shape::geom::properties::face_area;
//! ```

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
