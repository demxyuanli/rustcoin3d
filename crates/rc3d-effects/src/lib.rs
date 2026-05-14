//! Composable rendering effects configuration for rustcoin3d.
//!
//! This crate provides a declarative API for configuring the render pipeline,
//! with automatic dependency resolution between effects.
//!
//! # Example
//! ```ignore
//! use rc3d_effects::{RenderConfig, Shadow, PostEffect};
//!
//! let config = RenderConfig::new()
//!     .enable(Shadow::CSM { cascade_count: 4, resolution: 2048, soft: true })
//!     .enable(PostEffect::SSAO)
//!     .enable(PostEffect::Tonemap)
//!     .build();
//! ```

pub mod effect_graph;
pub mod post_effect;
pub mod render_config;
pub mod shadow;

pub use effect_graph::*;
pub use post_effect::*;
pub use render_config::*;
pub use shadow::*;
