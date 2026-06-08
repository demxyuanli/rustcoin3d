//! Self-contained 2D Delaunay triangulation engine.
//!
//! Inspired by Open CASCADE's BRepMesh_Delaun (Watson algorithm) but
//! implemented from scratch with Rust-idiomatic data structures and
//! several optimizations:
//!
//! - Bowyer-Watson incremental point insertion with super-triangle
//! - Cell-based circumcircle spatial index for O(1) conflict detection
//! - Constrained Delaunay (CDT) via edge splitting + flip-based enforcement
//! - Post-process edge-flip optimization for mesh quality
//! - Robust geometric predicates (orientation, in-circle)
//!
//! Alternative backend: DelaBella (Newton Apple Wrapper) — see `delabella/`.

mod geom;
mod half_edge;
mod circle_index;
mod triangulation;
mod cdt_adapter;
mod polygon_mesh;
pub(crate) mod delabella;

pub use geom::{Point2d, robust_orient2d, robust_in_circle};
pub use triangulation::{Delaunay2d, DelaunayConfig, Triangle};
pub use cdt_adapter::{NativeCdt, CdtVertHandle, DelaunayBackend, insert_uv_native, uv_bbox_from_loops};
pub use polygon_mesh::earcut_uv_polygon;
