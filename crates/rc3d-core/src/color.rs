use crate::math::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrgbColor(pub Vec3);

#[derive(Clone, Copy, Debug, PartialEq)]
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
