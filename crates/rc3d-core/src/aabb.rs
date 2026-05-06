use crate::math::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::MAX),
            max: Vec3::splat(f32::MIN),
        }
    }

    pub fn from_point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }

    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn transform(&self, matrix: Mat4) -> Aabb {
        let corners = [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ];
        let mut result = Aabb::empty();
        for c in &corners {
            let t = matrix.transform_point3(*c);
            result = result.union(&Aabb::from_point(t));
        }
        result
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Returns true if this AABB overlaps with another AABB.
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let a = Aabb::empty();
        assert!(a.min.x > a.max.x);
    }

    #[test]
    fn test_from_point() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let a = Aabb::from_point(p);
        assert_eq!(a.min, p);
        assert_eq!(a.max, p);
    }

    #[test]
    fn test_union_disjoint() {
        let a = Aabb::from_point(Vec3::new(0.0, 0.0, 0.0));
        let b = Aabb::from_point(Vec3::new(2.0, 3.0, 4.0));
        let u = a.union(&b);
        assert_eq!(u.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(u.max, Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn test_union_overlapping() {
        let a = Aabb { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(2.0, 2.0, 2.0) };
        let b = Aabb { min: Vec3::new(1.0, 1.0, 1.0), max: Vec3::new(3.0, 3.0, 3.0) };
        let u = a.union(&b);
        assert_eq!(u.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(u.max, Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_center() {
        let a = Aabb { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(2.0, 4.0, 6.0) };
        let c = a.center();
        assert_eq!(c, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_size() {
        let a = Aabb { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(2.0, 3.0, 4.0) };
        assert_eq!(a.size(), Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn test_transform_preserves_union() {
        use crate::math::Mat4;
        let a = Aabb::from_point(Vec3::ZERO);
        let b = Aabb::from_point(Vec3::new(1.0, 1.0, 1.0));
        let u = a.union(&b);
        let t = u.transform(Mat4::IDENTITY);
        assert_eq!(t.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(t.max, Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_transform_translate() {
        use crate::math::Mat4;
        let a = Aabb { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(1.0, 1.0, 1.0) };
        let t = a.transform(Mat4::from_translation(Vec3::new(2.0, 3.0, 4.0)));
        assert!((t.min.x - 2.0).abs() < 0.001);
        assert!((t.max.x - 3.0).abs() < 0.001);
    }
}
