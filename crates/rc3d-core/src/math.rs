// ── f32 types (for GPU rendering) ──
pub use glam::{Affine3A, Mat4, Quat, Vec2, Vec3, Vec4};

// ── f64 types (for geometry kernel) ──
/// Geometry-kernel scalar type (f64, matching OCCT `Standard_Real`).
pub type Real = f64;
pub use glam::{DAffine3 as PAffine3, DMat4 as PMat4, DQuat as PQuat, DVec2 as PVec2, DVec3 as PVec3, DVec4 as PVec4};
