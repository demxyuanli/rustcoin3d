//! Shadow configuration types.

/// Cascaded Shadow Maps configuration.
#[derive(Clone, Debug)]
pub struct CsmConfig {
    /// Number of cascades (1–4).
    pub cascade_count: u32,
    /// Shadow map resolution per cascade (e.g. 2048 → 2048×2048).
    pub resolution: u32,
    /// Enable PCF soft shadows.
    pub soft: bool,
}

impl Default for CsmConfig {
    fn default() -> Self {
        Self { cascade_count: 4, resolution: 2048, soft: true }
    }
}

/// Shadow configuration variants.
#[derive(Clone, Debug)]
pub enum Shadow {
    /// Cascaded Shadow Maps for directional lights.
    CSM(CsmConfig),
    /// Disable all shadows.
    Off,
}

impl Default for Shadow {
    fn default() -> Self {
        Shadow::CSM(CsmConfig::default())
    }
}
