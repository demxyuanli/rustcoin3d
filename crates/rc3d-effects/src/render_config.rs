//! Render configuration builder.

use rc3d_core::DisplayMode;

use crate::effect_graph::EffectGraph;
use crate::post_effect::PostEffect;
use crate::shadow::Shadow;

/// Declarative render pipeline configuration.
///
/// Users describe *what* effects they want; `EffectGraph` resolves *how*
/// to order and configure the passes.
#[derive(Clone, Debug)]
pub struct RenderConfig {
    display_mode: DisplayMode,
    shadow: Shadow,
    post_effects: Vec<PostEffect>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            display_mode: DisplayMode::Shaded,
            shadow: Shadow::CSM(Default::default()),
            post_effects: Vec::new(),
        }
    }
}

impl RenderConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// High-quality preset: Shaded, 4-cascade CSM, SSAO+SSR+Bloom+TAA+Tonemap.
    pub fn high_quality() -> Self {
        Self {
            display_mode: DisplayMode::Shaded,
            shadow: Shadow::CSM(Default::default()),
            post_effects: vec![
                PostEffect::SSAO,
                PostEffect::SSR,
                PostEffect::Bloom,
                PostEffect::TAA,
                PostEffect::Tonemap,
            ],
        }
    }

    /// Set the display mode.
    pub fn display_mode(mut self, mode: DisplayMode) -> Self {
        self.display_mode = mode;
        self
    }

    /// Enable a shadow configuration.
    pub fn shadow(mut self, shadow: Shadow) -> Self {
        self.shadow = shadow;
        self
    }

    /// Enable a post-processing effect.
    pub fn enable(mut self, effect: PostEffect) -> Self {
        if !self.post_effects.contains(&effect) {
            self.post_effects.push(effect);
        }
        self
    }

    /// Disable a post-processing effect.
    pub fn disable(mut self, effect: PostEffect) -> Self {
        self.post_effects.retain(|e| *e != effect);
        self
    }

    /// Compile into an `EffectGraph`.
    pub fn build(self) -> EffectGraph {
        EffectGraph::compile(&self)
    }

    /// Get the configured display mode.
    pub fn get_display_mode(&self) -> DisplayMode {
        self.display_mode
    }

    /// The configured shadow settings.
    pub fn shadow_settings(&self) -> &Shadow {
        &self.shadow
    }

    /// The configured post effects.
    pub fn post_effects(&self) -> &[PostEffect] {
        &self.post_effects
    }
}
