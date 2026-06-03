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

mod geom;
mod half_edge;
mod circle_index;
mod triangulation;
mod cdt_adapter;
mod polygon_mesh;

pub use geom::{Point2d, robust_orient2d, robust_in_circle};
pub use triangulation::{Delaunay2d, DelaunayConfig, Triangle};
pub use cdt_adapter::{NativeCdt, CdtVertHandle, insert_uv_native, uv_bbox_from_loops};
pub use polygon_mesh::earcut_uv_polygon;
