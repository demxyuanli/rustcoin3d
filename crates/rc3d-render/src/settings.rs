use crate::ibl::IblPreset;
use rc3d_core::DisplayMode;

#[derive(Clone, Debug)]
#[derive(Default)]
pub struct RenderSettings {
    pub post_effect: PostEffectSettings,
    pub lighting: LightingSettings,
    pub display: DisplaySettings,
    pub performance: PerformanceSettings,
}

#[derive(Clone, Copy, Debug)]
pub struct PerformanceSettings {
    /// GPU compute culling (requires enable_gpu_culling call).
    pub gpu_culling: bool,
    /// Min object count to enable GPU culling.
    pub gpu_culling_threshold: u32,
    /// Parallel scene traversal using rayon.
    pub parallel_traversal: bool,
    /// Mesh pool capacity (0 = disabled).
    pub mesh_pool_capacity: usize,
    /// Max GPU bytes for mesh pool slots.
    pub mesh_pool_max_mb: u64,
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            gpu_culling: false,
            gpu_culling_threshold: 4096,
            parallel_traversal: false,
            mesh_pool_capacity: 0,
            mesh_pool_max_mb: 512,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PostEffectSettings {
    pub hdr: bool,
    pub taa: bool,
    pub motion_blur: bool,
    pub ssr: bool,
    pub color_grading: bool,
    pub dof: bool,
    pub volumetric_fog: bool,
    pub ldr_fxaa: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct LightingSettings {
    pub cluster_lights: bool,
    pub omni_shadows: bool,
    pub ibl_preset: IblPreset,
}

#[derive(Clone, Copy, Debug)]
pub struct DisplaySettings {
    pub display_mode: DisplayMode,
    pub grid_enabled: bool,
    pub hud_enabled: bool,
    pub outline_width: f32,
    pub outline_color: [f32; 4],
    pub xray_mode: bool,
    pub vsync_enabled: bool,
    pub screen_space_selection_outline: bool,
}


impl Default for PostEffectSettings {
    fn default() -> Self {
        Self {
            hdr: false,
            taa: false,
            motion_blur: false,
            ssr: false,
            color_grading: false,
            dof: false,
            volumetric_fog: false,
            ldr_fxaa: true,
        }
    }
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            cluster_lights: true,
            omni_shadows: true,
            ibl_preset: IblPreset::Studio,
        }
    }
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            display_mode: DisplayMode::ShadedWithEdges,
            grid_enabled: false,
            hud_enabled: true,
            outline_width: 1.0,
            outline_color: [1.0, 0.5, 0.0, 1.0],
            xray_mode: false,
            vsync_enabled: true,
            screen_space_selection_outline: true,
        }
    }
}
