//! High-level scene building API for rustcoin3d.
//!
//! This crate provides a declarative DSL for constructing 3D scenes
//! without touching the low-level SceneGraph/NodeData internals.
//!
//! # Example
//! ```ignore
//! use rc3d_scene_api::{Scene, Cube, Material};
//!
//! let mut scene = Scene::new();
//! scene.add(Cube::default()
//!     .at(1.0, 0.0, 0.0)
//!     .material(Material::pbr().base_color(0.8, 0.2, 0.2)));
//! let graph = scene.build();
//! ```

pub mod animation;
pub mod camera;
pub mod geometry;
pub mod group;
pub mod light;
pub mod material;
pub mod query;
pub mod scene;
pub mod shape;

pub use animation::*;
pub use camera::*;
pub use geometry::*;
pub use group::*;
pub use light::*;
pub use material::*;
pub use query::*;
pub use scene::*;
pub use shape::*;
