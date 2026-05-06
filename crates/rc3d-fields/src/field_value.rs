use rc3d_core::math::{Mat4, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FieldValue {
    Bool(bool),
    Int32(i32),
    Float(f32),
    Float64(f64),
    Vec2f(Vec2),
    Vec3f(Vec3),
    Vec4f(Vec4),
    Mat4f(Mat4),
    // Multi-value fields
    FloatArray(Vec<f32>),
    Vec3fArray(Vec<Vec3>),
    Int32Array(Vec<i32>),
    // Variable-length types
    String(String),
    Binary(Vec<u8>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float64_roundtrip() {
        let v = FieldValue::Float64(42.0);
        assert_eq!(v, FieldValue::Float64(42.0));
        assert_ne!(v, FieldValue::Float(42.0));
        assert_ne!(v, FieldValue::Float64(43.0));
    }

    #[test]
    fn test_string_roundtrip() {
        let v = FieldValue::String("hello".to_string());
        assert_eq!(v, FieldValue::String("hello".to_string()));
        assert_ne!(v, FieldValue::String("world".to_string()));
    }

    #[test]
    fn test_string_clone() {
        let v = FieldValue::String("part_number".to_string());
        assert_eq!(v.clone(), v);
    }

    #[test]
    fn test_binary_roundtrip() {
        let data = vec![0u8, 1, 2, 255];
        let v = FieldValue::Binary(data.clone());
        assert_eq!(v, FieldValue::Binary(data));
    }

    #[test]
    fn test_binary_empty() {
        let v = FieldValue::Binary(vec![]);
        assert_eq!(v, FieldValue::Binary(vec![]));
    }

    #[test]
    fn test_string_debug_format() {
        let v = FieldValue::String("test".into());
        let s = format!("{:?}", v);
        assert!(s.contains("String"));
        assert!(s.contains("test"));
    }

    #[test]
    fn test_binary_debug_format() {
        let v = FieldValue::Binary(vec![1, 2, 3]);
        let s = format!("{:?}", v);
        assert!(s.contains("Binary"));
    }
}
