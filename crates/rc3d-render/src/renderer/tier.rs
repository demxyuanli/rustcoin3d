//! CAD display tier apply + compositor CAD look override.

use rc3d_core::DisplayMode;

use super::internals::{CadDisplayTier, GpuTier, TierConfig};

pub(super) struct CadLookSnapshot {
    enable_bloom: bool,
    bloom_str: f32,
    xray_mode: bool,
    grid_enabled: bool,
    global_display_mode: DisplayMode,
}

impl super::Renderer {
    /// Set the CAD display quality tier. Clamped by GPU capability on Basic tier.
    pub fn set_display_tier(&mut self, tier: CadDisplayTier) {
        self.cad_tier_authoritative = true;
        let max_tier = match self.gpu.gpu_capability.tier {
            GpuTier::Basic => CadDisplayTier::Visualization,
            GpuTier::Standard => CadDisplayTier::IndustrialDisplay,
            GpuTier::Enhanced => CadDisplayTier::ProductRendering,
        };
        let requested = if tier > max_tier { max_tier } else { tier };
        if tier != requested {
            log::warn!(
                "Requested tier {:?} exceeds GPU capability (max {:?}); clamping to {:?}",
                tier, max_tier, requested
            );
        }
        if self.gpu.requested_tier != requested {
            self.gpu.requested_tier = requested;
            self.gpu.effective_tier = requested;
            self.apply_tier_config(true);
        } else if self.gpu.effective_tier != requested {
            // Effective can lag behind requested during orbit degrade/recovery; choosing the
            // same tier again (or repeating a tier hotkey) must snap visuals without changing request.
            self.gpu.effective_tier = requested;
            self.gpu.tier_cooldown_frames = 0;
            self.apply_tier_config(true);
        }
    }

    /// Called once per frame. Updates effective tier with degradation and recovery.
    /// Reads `self.interaction_active` (set by app during camera orbit/pan/zoom).
    pub fn update_tier(&mut self) {
        let requested = self.gpu.requested_tier;
        let max_tier = match self.gpu.gpu_capability.tier {
            GpuTier::Basic => CadDisplayTier::Visualization,
            GpuTier::Standard => CadDisplayTier::IndustrialDisplay,
            GpuTier::Enhanced => CadDisplayTier::ProductRendering,
        };
        let clamped = if requested > max_tier { max_tier } else { requested };

        // Detect interaction stop: start cooldown for recovery
        if self.gpu.interaction_active && !self.interaction_active {
            self.gpu.tier_cooldown_frames = 30; // ~500ms at 60fps
        }
        self.gpu.interaction_active = self.interaction_active;

        // Degrade during interaction (tiers 2+ only)
        if self.interaction_active {
            let degraded = match clamped {
                CadDisplayTier::ProductRendering => CadDisplayTier::Visualization,
                CadDisplayTier::IndustrialDisplay => CadDisplayTier::Visualization,
                other => other,
            };
            if self.gpu.effective_tier != degraded {
                self.gpu.effective_tier = degraded;
                self.gpu.tier_cooldown_frames = 0;
                self.apply_tier_config(false);
            }
            return;
        }

        // Snap down when effective exceeds allowed target (requested lowered or clamp tightened).
        if self.gpu.effective_tier > clamped {
            self.gpu.effective_tier = clamped;
            self.gpu.tier_cooldown_frames = 0;
            self.apply_tier_config(false);
        }

        // Recovery: step up one tier per cooldown period
        if self.gpu.effective_tier < clamped {
            if self.gpu.tier_cooldown_frames > 0 {
                self.gpu.tier_cooldown_frames -= 1;
            } else {
                let next = self.gpu.effective_tier as u32 + 1;
                let next_tier = CadDisplayTier::from_u32(next);
                if next_tier <= clamped {
                    self.gpu.effective_tier = next_tier;
                    self.gpu.tier_cooldown_frames = 30;
                    self.apply_tier_config(false);
                }
            }
        }
    }

    /// Apply feature toggles for the current effective tier.
    ///
    /// When `user_initiated` is true (user changed tier via UI), the tier's
    /// display-mode semantics are also applied (e.g. DesignCreation → Flat).
    /// When false (automatic degradation/recovery), display mode is preserved
    /// so the user's explicit choice is not overridden.
    pub(super) fn apply_tier_config(&mut self, user_initiated: bool) {
        let cfg = TierConfig::for_tier(self.gpu.effective_tier);
        log::debug!(
            "Tier config applied: {:?} (user={}) | HDR={} SSAO={} TAA={} SSR={} DOF={} Fog={}",
            self.gpu.effective_tier, user_initiated,
            cfg.hdr_post, cfg.ssao, cfg.taa, cfg.ssr, cfg.dof, cfg.volumetric_fog
        );
        // Apply display-mode semantics on user-initiated tier switch.
        // Entering a flat-shading tier forces Flat; leaving restores user's choice.
        if user_initiated {
            self.global_display_mode = if cfg.flat_shading {
                if cfg.edges { DisplayMode::FlatWithEdge } else { DisplayMode::Flat }
            } else if cfg.edges {
                DisplayMode::ShadedWithEdges
            } else {
                DisplayMode::Shaded
            };
        }
        self.tier_wants_shadow = cfg.shadows;
        self.tier_wants_edges = cfg.edges;
        self.enable_taa = cfg.taa;
        self.enable_ldr_fxaa = cfg.fxaa && !cfg.taa;
        self.enable_ssao = cfg.ssao;
        self.enable_bloom = false;
        self.enable_motion_blur = cfg.motion_blur && self.gpu.motion_blur.is_some();
        self.enable_ssr = cfg.ssr;
        self.enable_color_grading = cfg.color_grading;
        self.enable_dof = cfg.dof;
        self.enable_volumetric_fog = cfg.volumetric_fog;
        self.hdr_post_processing = cfg.hdr_post;
        if self.hdr_post_processing && self.gpu.post_fx.is_none() {
            self.ensure_post_fx_targets();
        }
        if self.enable_motion_blur && !self.hdr_post_processing {
            self.enable_motion_blur = false;
        }
        if self.enable_motion_blur {
            self.enable_taa = true;
        }
    }

    /// Flat-shading tiers (e.g. DesignCreation) always render as Flat; display-mode changes
    /// still update `user_display_mode` so leaving the tier restores the user's choice.
    pub(super) fn clamp_global_display_mode_for_flat_shading_tier(&mut self) {
        let cfg = TierConfig::for_tier(self.gpu.effective_tier);
        if cfg.flat_shading {
            // Wireframe is an explicit user choice; don't overwrite it.
            if self.global_display_mode == DisplayMode::Wireframe {
                return;
            }
            self.global_display_mode = if cfg.edges {
                DisplayMode::FlatWithEdge
            } else {
                DisplayMode::Flat
            };
        }
    }

    /// Re-apply pipeline toggles for the current effective CAD tier (used when tier is
    /// authoritative so interaction overrides / inspector cannot bypass CAD tier).
    pub fn reapply_cad_tier_constraints(&mut self) {
        self.apply_tier_config(false);
        self.clamp_global_display_mode_for_flat_shading_tier();
    }

    /// Apply compositor-graph CAD look after tier constraints.
    /// Inactive graphs restore the pre-look snapshot so menus/tier stay authoritative.
    pub fn apply_compositor_cad_look(&mut self) {
        let look = self.compositor_graph.cad_look();
        if !look.active {
            if let Some(s) = self.cad_look_snapshot.take() {
                self.enable_bloom = s.enable_bloom;
                self.post_fx_params.bloom_str = s.bloom_str;
                self.upload_post_fx_params();
                self.xray_mode = s.xray_mode;
                self.grid_enabled = s.grid_enabled;
                self.global_display_mode = s.global_display_mode;
            }
            return;
        }
        if self.cad_look_snapshot.is_none() {
            self.cad_look_snapshot = Some(CadLookSnapshot {
                enable_bloom: self.enable_bloom,
                bloom_str: self.post_fx_params.bloom_str,
                xray_mode: self.xray_mode,
                grid_enabled: self.grid_enabled,
                global_display_mode: self.global_display_mode,
            });
        }
        if look.needs_hdr() {
            self.hdr_post_processing = true;
            if self.gpu.post_fx.is_none() {
                self.ensure_post_fx_targets();
            }
        } else {
            self.hdr_post_processing = false;
        }
        self.enable_ssao = look.ssao;
        self.enable_taa = look.taa;
        self.enable_ldr_fxaa = look.fxaa && !look.taa;
        self.enable_color_grading = look.color_grading;
        self.enable_bloom = look.bloom;
        if look.bloom {
            self.post_fx_params.bloom_str = look.bloom_str.max(0.05);
        } else {
            self.post_fx_params.bloom_str = 0.0;
        }
        self.upload_post_fx_params();
        self.enable_dof = look.dof;
        self.enable_ssr = look.ssr;
        self.enable_volumetric_fog = look.fog;
        self.xray_mode = look.xray;
        self.grid_enabled = look.grid;
        self.tier_wants_shadow = look.shadows;
        self.tier_wants_edges = look.edges || look.hidden_line;
        if look.hidden_line {
            self.global_display_mode = DisplayMode::HiddenLine;
        } else if look.edges {
            self.global_display_mode = DisplayMode::ShadedWithEdges;
        } else {
            self.global_display_mode = DisplayMode::Shaded;
        }
    }

    /// Whether CAD tier was explicitly chosen via [`Self::set_display_tier`].
    pub fn cad_tier_authoritative(&self) -> bool {
        self.cad_tier_authoritative
    }

    /// Compact HUD line for Studio CAD-look verification.
    pub fn cad_status_line(&self) -> String {
        fn tier_name(tier: CadDisplayTier) -> &'static str {
            match tier {
                CadDisplayTier::DesignCreation => "Design",
                CadDisplayTier::Visualization => "Viz",
                CadDisplayTier::IndustrialDisplay => "Industrial",
                CadDisplayTier::ProductRendering => "Product",
            }
        }
        let req = tier_name(self.gpu.requested_tier);
        let eff = tier_name(self.gpu.effective_tier);
        let look = self.compositor_graph.cad_look();
        let src = if look.active { "comp" } else { "tier" };
        let tier = if req == eff {
            eff.to_string()
        } else {
            format!("{eff}<-{req}")
        };
        let mut flags: Vec<&str> = Vec::new();
        if self.hdr_post_processing {
            flags.push("HDR");
        }
        if self.enable_ssao {
            flags.push("SSAO");
        }
        if self.enable_taa {
            flags.push("TAA");
        }
        if self.enable_ldr_fxaa {
            flags.push("FXAA");
        }
        if self.enable_color_grading {
            flags.push("CG");
        }
        if self.enable_bloom {
            flags.push("Bloom");
        }
        if self.enable_dof {
            flags.push("DOF");
        }
        if self.enable_ssr {
            flags.push("SSR");
        }
        if self.enable_volumetric_fog {
            flags.push("Fog");
        }
        if self.xray_mode {
            flags.push("XRay");
        }
        if self.grid_enabled {
            flags.push("Grid");
        }
        if self.tier_wants_shadow {
            flags.push("Shadow");
        }
        if self.global_display_mode == DisplayMode::HiddenLine {
            flags.push("HLR");
        } else if self.tier_wants_edges {
            flags.push("Edges");
        }
        if flags.is_empty() {
            format!("CAD: {tier}/{src}")
        } else {
            format!("CAD: {tier}/{src} {}", flags.join(" "))
        }
    }
}
