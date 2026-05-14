//! Effect graph: DAG of ordered render passes with dependency resolution.

use std::collections::HashSet;

use rc3d_render::CSM_CASCADE_COUNT;

use crate::post_effect::PostEffect;
use crate::render_config::RenderConfig;
use crate::shadow::Shadow;

/// Compiled effect graph, ready to be applied to a `PassContext`.
#[derive(Clone, Debug)]
pub struct EffectGraph {
    pub run_shadow_pass: bool,
    pub cascade_count: u32,
    pub shadow_map_size: u32,
    pub soft_shadows: bool,

    pub enable_ssao: bool,
    pub enable_ssr: bool,
    pub enable_bloom: bool,
    pub enable_taa: bool,
    pub enable_tonemap: bool,
    pub enable_dof: bool,
    pub enable_motion_blur: bool,
    pub enable_volumetric_fog: bool,

    /// Derived: SSR requires HZB.
    pub enable_hzb: bool,
    /// Derived: Bloom or Tonemap requires HDR.
    pub enable_hdr: bool,
}

impl EffectGraph {
    /// Compile a RenderConfig into an ordered effect graph.
    pub(crate) fn compile(config: &RenderConfig) -> Self {
        let mut g = EffectGraph {
            run_shadow_pass: false,
            cascade_count: 1,
            shadow_map_size: 1,
            soft_shadows: false,
            enable_ssao: false,
            enable_ssr: false,
            enable_bloom: false,
            enable_taa: false,
            enable_tonemap: false,
            enable_dof: false,
            enable_motion_blur: false,
            enable_volumetric_fog: false,
            enable_hzb: false,
            enable_hdr: false,
        };

        // Shadow
        match config.shadow_settings() {
            Shadow::CSM(csm) => {
                g.run_shadow_pass = true;
                g.cascade_count = csm.cascade_count.min(CSM_CASCADE_COUNT as u32);
                g.shadow_map_size = csm.resolution;
                g.soft_shadows = csm.soft;
            }
            Shadow::Off => {}
        }

        // Post effects
        let effect_set: HashSet<_> = config.post_effects().iter().collect();
        g.enable_ssao = effect_set.contains(&PostEffect::SSAO);
        g.enable_ssr = effect_set.contains(&PostEffect::SSR);
        g.enable_bloom = effect_set.contains(&PostEffect::Bloom);
        g.enable_taa = effect_set.contains(&PostEffect::TAA);
        g.enable_tonemap = effect_set.contains(&PostEffect::Tonemap);
        g.enable_dof = effect_set.contains(&PostEffect::DOF);
        g.enable_motion_blur = effect_set.contains(&PostEffect::MotionBlur);
        g.enable_volumetric_fog = effect_set.contains(&PostEffect::VolumetricFog);

        // Automatic dependency resolution
        g.enable_hzb = g.enable_ssr; // SSR requires HZB
        g.enable_hdr = g.enable_bloom || g.enable_tonemap;

        g
    }

    /// Apply this effect graph to a render pass context.
    ///
    /// Sets the relevant boolean flags on the `PassContext`.
    /// Call this before each frame's render pass setup.
    pub fn apply_booleans(&self) -> EffectBooleans {
        EffectBooleans {
            run_shadow_pass: self.run_shadow_pass,
            enable_ssao: self.enable_ssao,
            enable_ssr: self.enable_ssr,
            enable_bloom: self.enable_bloom,
            enable_taa: self.enable_taa,
            enable_tonemap: self.enable_tonemap,
            enable_dof: self.enable_dof,
            enable_motion_blur: self.enable_motion_blur,
            enable_volumetric_fog: self.enable_volumetric_fog,
            enable_hzb: self.enable_hzb,
            enable_hdr: self.enable_hdr,
        }
    }
}

/// Flat struct of booleans for applying to PassContext.
#[derive(Clone, Debug, Default)]
pub struct EffectBooleans {
    pub run_shadow_pass: bool,
    pub enable_ssao: bool,
    pub enable_ssr: bool,
    pub enable_bloom: bool,
    pub enable_taa: bool,
    pub enable_tonemap: bool,
    pub enable_dof: bool,
    pub enable_motion_blur: bool,
    pub enable_volumetric_fog: bool,
    pub enable_hzb: bool,
    pub enable_hdr: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_no_post_effects() {
        let config = RenderConfig::default();
        let graph = config.build();
        assert!(graph.run_shadow_pass);
        assert!(!graph.enable_ssao);
        assert!(!graph.enable_hzb);
        assert!(!graph.enable_hdr);
    }

    #[test]
    fn test_high_quality_config() {
        let config = RenderConfig::high_quality();
        let graph = config.build();
        assert!(graph.run_shadow_pass);
        assert!(graph.enable_ssao);
        assert!(graph.enable_ssr);
        assert!(graph.enable_hzb); // derived from SSR
        assert!(graph.enable_bloom);
        assert!(graph.enable_taa);
        assert!(graph.enable_tonemap);
        assert!(graph.enable_hdr); // derived from Bloom/Tonemap
    }

    #[test]
    fn test_no_shadows() {
        let config = RenderConfig::new().shadow(Shadow::Off);
        let graph = config.build();
        assert!(!graph.run_shadow_pass);
        assert_eq!(graph.cascade_count, 1);
    }

    #[test]
    fn test_ssr_requires_hzb() {
        let config = RenderConfig::new().enable(PostEffect::SSR);
        let graph = config.build();
        assert!(graph.enable_hzb);
    }

    #[test]
    fn test_selective_effects() {
        let config = RenderConfig::new()
            .shadow(Shadow::Off)
            .enable(PostEffect::TAA)
            .enable(PostEffect::Tonemap);
        let graph = config.build();
        assert!(!graph.run_shadow_pass);
        assert!(!graph.enable_ssao);
        assert!(graph.enable_taa);
        assert!(graph.enable_tonemap);
        assert!(graph.enable_hdr); // Tonemap requires HDR
    }
}
