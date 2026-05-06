use crate::math::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SrgbColor(pub Vec3);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearColor(pub Vec3);

impl SrgbColor {
    pub fn to_linear(self) -> LinearColor {
        fn linearize(c: f32) -> f32 {
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }
        LinearColor(Vec3::new(
            linearize(self.0.x),
            linearize(self.0.y),
            linearize(self.0.z),
        ))
    }
}

impl LinearColor {
    pub fn to_srgb(self) -> SrgbColor {
        fn delinearize(c: f32) -> f32 {
            if c <= 0.0031308 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            }
        }
        SrgbColor(Vec3::new(
            delinearize(self.0.x),
            delinearize(self.0.y),
            delinearize(self.0.z),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_roundtrip() {
        let srgb = SrgbColor(Vec3::new(0.0, 0.0, 0.0));
        let linear = srgb.to_linear();
        let back = linear.to_srgb();
        assert!((back.0.x).abs() < 0.001);
        assert!((back.0.y).abs() < 0.001);
    }

    #[test]
    fn test_white_roundtrip() {
        let srgb = SrgbColor(Vec3::new(1.0, 1.0, 1.0));
        let linear = srgb.to_linear();
        let back = linear.to_srgb();
        assert!((back.0.x - 1.0).abs() < 0.01);
        assert!((back.0.y - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_to_srgb_clamped() {
        // Negative values should produce something sane, not NaN
        let linear = LinearColor(Vec3::new(-0.1, 0.5, 1.5));
        let srgb = linear.to_srgb();
        assert!(srgb.0.x.is_finite());
        assert!(srgb.0.y.is_finite());
        assert!(srgb.0.z.is_finite());
    }

}
