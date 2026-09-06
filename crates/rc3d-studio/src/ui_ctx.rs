use std::collections::HashSet;

use rc3d_core::DisplayMode;
use rc3d_editor::{EditorInteractionState, EditorSession, EditorUiContext, RenderFeatureFlags};
use rc3d_engine_api::Engine;
use rc3d_render::viewport::LayoutMode;

use crate::cases::{self, ActiveCase};

pub fn build_ui_ctx(
    engine: &Engine,
    session: &EditorSession,
    interaction: &EditorInteractionState,
    active_case: Option<&ActiveCase>,
) -> EditorUiContext {
    let selected: HashSet<_> = engine.world.graph.selected_nodes().iter().copied().collect();
    let selected_count = selected.len();
    let r = engine.renderer.as_ref();
    let display_mode = r.map(|r| r.display_mode()).unwrap_or(DisplayMode::Shaded);
    let layout = r.map(|r| r.viewport_layout().layout_mode).unwrap_or(LayoutMode::Single);
    let active_vp = r
        .and_then(|r| r.viewport_layout().active())
        .map(|v| v.name.clone())
        .unwrap_or_else(|| "Perspective".into());
    let (case_params, case_steps, active_case_id) = match active_case {
        Some(a) => (
            cases::param_views(a),
            cases::step_views(a),
            Some(a.id.clone()),
        ),
        None => (Vec::new(), Vec::new(), None),
    };
    EditorUiContext {
        selected,
        display_mode_label: format!("{display_mode:?}"),
        ibl_label: r.map(|r| r.ibl_preset_name().to_string()).unwrap_or_default(),
        ibl_preset: r.map(|r| r.ibl_preset).unwrap_or(rc3d_render::ibl::IblPreset::Studio),
        gizmo_mode: engine.gizmo.mode,
        layout_mode: layout,
        viewport_h_split: r.map(|r| r.viewport_layout().quad_h_split).unwrap_or(0.5),
        viewport_v_split: r.map(|r| r.viewport_layout().quad_v_split).unwrap_or(0.5),
        viewport_split_hover: interaction.view_split_hover,
        viewport_split_drag: interaction.view_split_drag,
        layout_mode_label: format!("{layout:?}"),
        active_viewport_label: active_vp,
        smoothed_fps: engine.fps.smoothed_fps(),
        frame_time_ms: engine.fps.average_frame_ms(),
        diagnostics: r.and_then(|r| r.last_diagnostics()).cloned(),
        hidden_nodes: engine.hidden_nodes.clone(),
        render_features: r
            .map(|r| RenderFeatureFlags {
                taa: r.enable_taa,
                motion_blur: r.enable_motion_blur,
                ssr: r.enable_ssr,
                color_grading: r.enable_color_grading,
                dof: r.enable_dof,
                volumetric_fog: r.enable_volumetric_fog,
                cluster_lights: r.enable_cluster_lights,
                omni_shadows: r.enable_omni_shadows,
                ldr_fxaa: r.enable_ldr_fxaa,
                screen_space_edges: r.screen_space_edges,
                screen_space_selection_outline: r.screen_space_selection_outline,
                gpu_cull: r.gpu_culling_enabled(),
                parallel_traversal: r.parallel_traversal_enabled(),
            })
            .unwrap_or_default(),
        hdr_enabled: r.map(|r| r.hdr_post_processing).unwrap_or(false),
        vsync_enabled: r.map(|r| r.vsync_enabled()).unwrap_or(true),
        grid_enabled: r.map(|r| r.grid_enabled).unwrap_or(false),
        hud_enabled: r.map(|r| r.hud_enabled).unwrap_or(true),
        outline_width: r.map(|r| r.outline_width).unwrap_or(1.0),
        outline_color: r.map(|r| r.outline_color).unwrap_or([1.0, 0.5, 0.0, 1.0]),
        xray_mode: r.map(|r| r.xray_mode).unwrap_or(false),
        ghost_unselected: r.map(|r| r.ghost_unselected).unwrap_or(false),
        ghost_opacity: r.map(|r| r.ghost_opacity).unwrap_or(0.18),
        feature_edge_color: r.map(|r| r.feature_edge_color).unwrap_or([0.9, 0.15, 0.1, 1.0]),
        wireframe_edge_color: r
            .map(|r| r.wireframe_edge_color)
            .unwrap_or([0.1, 0.2, 0.55, 1.0]),
        hidden_edge_color: r.map(|r| r.hidden_edge_color).unwrap_or([0.45, 0.45, 0.45, 1.0]),
        crease_angle: rc3d_render::feature_crease_angle(),
        ss_edge_threshold: r.map(|r| r.ss_edge_threshold).unwrap_or(0.015),
        wboit_enabled: r.map(|r| r.enable_wboit).unwrap_or(true),
        adaptive_quality_mode: session.adaptive_quality_mode,
        adaptive_quality_name: r
            .map(|r| r.adaptive_quality_name().to_string())
            .unwrap_or_else(|| "High".into()),
        cad_display_tier: r
            .map(|r| r.display_tier())
            .unwrap_or(rc3d_render::renderer::CadDisplayTier::Visualization),
        bookmarks: {
            let mut slots = [(false, "1"); 9];
            let names = ["1", "2", "3", "4", "5", "6", "7", "8", "9"];
            for i in 0..9 {
                slots[i] = (engine.controller.bookmarks[i].is_some(), names[i]);
            }
            slots
        },
        selected_count,
        document_path: session.document_path.clone(),
        document_dirty: session.dirty,
        bg_mode: session.background.mode,
        bg_top: session.background.top_color,
        bg_bot: session.background.bot_color,
        post_vignette: r.map(|r| r.post_fx_params.vignette).unwrap_or(0.3),
        post_chromatic: r.map(|r| r.post_fx_params.chromatic).unwrap_or(0.0),
        post_bloom: r.map(|r| r.post_fx_params.bloom_str).unwrap_or(0.8),
        post_grain: r.map(|r| r.post_fx_params.grain).unwrap_or(0.0),
        post_halftone: r.map(|r| r.post_fx_params.halftone).unwrap_or(0.0),
        post_glitch: r.map(|r| r.post_fx_params.glitch).unwrap_or(0.0),
        csm_resolution: r.map(|r| r.csm_shadow_params().0).unwrap_or(2048),
        csm_cascades: r.map(|r| r.csm_shadow_params().1).unwrap_or(4),
        interaction_render_scale: r.map(|r| r.interaction_render_scale).unwrap_or(1.0),
        walk_mode: engine
            .viewport_cameras
            .active()
            .map(|vc| vc.controller.walk_mode)
            .unwrap_or(engine.controller.walk_mode),
        camera_from: {
            let c = engine
                .viewport_cameras
                .active()
                .map(|vc| &vc.controller)
                .unwrap_or(&engine.controller);
            (c.eye_position() - c.target)
                .normalize_or_zero()
                .to_array()
        },
        camera_up: {
            let c = engine
                .viewport_cameras
                .active()
                .map(|vc| &vc.controller)
                .unwrap_or(&engine.controller);
            c.up.to_array()
        },
        ui_theme: session.ui_theme,
        ui_locale: session.ui_locale,
        select_kind: interaction.select_kind,
        canvas_tool: interaction.canvas,
        measurement_type: interaction.measurement_type,
        markup_tool: interaction.markup.tool,
        measure_label: interaction.measure_label.clone(),
        measure_points: interaction.measure.points.len(),
        measure_needed: interaction.measure.points_needed(),
        keymap: session.keymap.clone(),
        locked_nodes: interaction.locked_nodes.clone(),
        history_undo: session.history.undo_log(),
        history_redo: session.history.redo_log(),
        file_dialog_dir: session.file_dialog_dir.clone(),
        case_catalog: cases::catalog_list(),
        active_case_id,
        case_params,
        case_steps,
    }
}
