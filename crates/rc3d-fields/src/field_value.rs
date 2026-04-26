use rc3d_core::math::{Mat4, Vec2, Vec3, Vec4};

#[derive(Clone, Debug, PartialEq)]
pub enum FieldValue {
    Bool(bool),
    Int32(i32),
    Float(f32),
    Vec2f(Vec2),
    Vec3f(Vec3),
    Vec4f(Vec4),
    Mat4f(Mat4),
    // Multi-value fields
    FloatArray(Vec<f32>),
    Vec3fArray(Vec<Vec3>),
    Int32Array(Vec<i32>),
}
