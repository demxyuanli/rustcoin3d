//! Re-export B-Rep store from `rc3d-shape`.
pub use rc3d_shape::store::{BRepStore, PCurveEdit};
#[deprecated(note = "use rc3d_shape::BRepStore instead")]
pub use rc3d_shape::store::BRepRegistry;
