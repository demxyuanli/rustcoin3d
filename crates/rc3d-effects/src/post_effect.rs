//! Post-processing effect types.

/// A post-processing effect that can be enabled in the render pipeline.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PostEffect {
    /// Screen-space ambient occlusion.
    SSAO,
    /// Screen-space reflections (requires HZB).
    SSR,
    /// Bloom (bright-pass → blur → composite).
    Bloom,
    /// Temporal anti-aliasing.
    TAA,
    /// HDR-to-LDR tone mapping.
    Tonemap,
    /// Depth of field.
    DOF,
    /// Per-pixel motion blur.
    MotionBlur,
    /// Volumetric fog / atmospheric scattering.
    VolumetricFog,
}

impl PostEffect {
    /// Returns true if this effect requires HZB.
    pub fn requires_hzb(&self) -> bool {
        matches!(self, PostEffect::SSR)
    }

    /// Returns true if this effect requires HDR rendering.
    pub fn requires_hdr(&self) -> bool {
        matches!(self, PostEffect::Bloom | PostEffect::Tonemap)
    }
}
