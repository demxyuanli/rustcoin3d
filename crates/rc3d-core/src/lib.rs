//! # rc3d-core — Shared Kernel
//!
//! Foundation types and utilities used across all rustcoin3d crates.
//! No dependency on scene graph, rendering, or I/O.
//!
//! ## Architecture
//! - `math` — `Real` (f32/f64), `PVec3` (glam), geometric primitives
//! - `aabb` — Axis-aligned bounding box with union/intersection/ray queries
//! - `bvh` — Bounding volume hierarchy for spatial acceleration
//! - `color` — Linear/sRGB color types (`Color4f`, conversions)
//! - `display` — `DisplayMode` presets, `FillStyle` / `EdgeStyle`, named `VisualStyle` catalog, `ClipPlane`
//! - `id` — `NodeId`, `GeometryId`, `MeshId` slot-map keys
//! - `projection` — Camera projection matrix builders (`depth_reversed_z_from_projection`)
//! - `error` — `EngineError` / `EngineResult` shared error types
//! - `utils` — Graph algorithms (toposort, BFS), float hashing, ring buffer, sort helpers
//!
//! ## OCC alignment
//! Corresponds to OpenCASCADE `gp`, `Bnd`, `BVH`, `Quantity`, and `TCollection` layers.
//!
//! ## Usage
//! ```ignore
//! use rc3d_core::math::{Real, PVec3};
//! use rc3d_core::aabb::Aabb;
//! ```

pub mod aabb;
pub mod bvh;
pub mod color;
pub mod display;
pub mod error;
pub mod id;
pub mod math;
pub mod projection;
pub mod utils;

pub use aabb::*;
pub use bvh::*;
pub use color::*;
pub use display::*;
pub use error::{EngineError, EngineResult};
pub use id::*;
pub use projection::depth_reversed_z_from_projection;
