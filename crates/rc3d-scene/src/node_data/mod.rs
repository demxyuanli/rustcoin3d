//! Scene graph node payloads (cameras, geometry, lights).
//!
//! ## Depth convention (`PerspectiveCameraNode` / `OrthographicCameraNode`)
//!
//! Set [`PerspectiveCameraNode::reverse_depth`] or [`OrthographicCameraNode::reverse_depth`] to build
//! a projection whose clip-space Z ordering matches **reverse-Z** rendering (WebGPU:
//! `CompareFunction::Greater`, depth clear `0.0`, HZB max pyramid). Leave `false` for **forward-Z**
//! (`Less`, clear `1.0`, HZB min pyramid).
//!
//! `DrawCall::depth_reversed_z` (see `rc3d-render`) is derived from the active projection matrix via
//! `rc3d_core::depth_reversed_z_from_projection`; keep camera projection and GPU depth state aligned.
//! If one frame mixes draw calls built from incompatible projections, `render_draw_calls` logs a
//! warning and uses the first visible draw call for pipeline depth mode.

mod advanced;
mod cameras;
mod control;
mod effects;
mod enum_data;
mod grouping;
mod lights;
mod properties;
mod shapes;

pub use advanced::*;
pub use cameras::*;
pub use control::*;
pub use effects::*;
pub use enum_data::*;
pub use grouping::*;
pub use lights::*;
pub use properties::*;
pub use shapes::*;
