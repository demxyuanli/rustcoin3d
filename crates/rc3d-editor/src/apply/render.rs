use rc3d_core::{DisplayMode, NodeId};
use rc3d_engine_api::Engine;
use rc3d_render::AdaptiveControl;
use rc3d_scene::SceneGraph;

use crate::commands::{AdaptiveQualityMode, EditorCommand};
use crate::context::EditorInteractionState;
use crate::ui::types::EditorDisplayMode;

use super::EditorSession;

pub(super) fn apply(
    engine: &mut Engine,
    _interaction: &mut EditorInteractionState,
    session: &mut EditorSession,
    cmd: EditorCommand,
) {
    match cmd {
        EditorCommand::ApplyVisualStyle(name) => {
            engine.apply_visual_style_to_selected(&name);
        }
        EditorCommand::SetViewportLayoutMode(mode) => {
            engine.set_layout_mode(mode);
        }
        EditorCommand::CycleViewportLayout => {
            engine.cycle_layout_mode();
        }
        EditorCommand::CycleActiveViewport => {
            engine.cycle_active_viewport();
        }
        EditorCommand::SetViewPreset(preset) => {
            engine.set_view_preset(preset);
        }
        EditorCommand::SetViewFromDirection(from) => {
            engine.set_view_from_direction(rc3d_core::math::Vec3::from_array(from));
        }
        EditorCommand::OrbitView { dx, dy } => {
            engine.orbit_view(dx, dy);
        }
        EditorCommand::SetWboit(enabled) => {
            engine.set_wboit(enabled);
        }
        EditorCommand::SetXrayMode(enabled) => {
            engine.set_xray_mode(enabled);
        }
        EditorCommand::SetGhostUnselected(enabled) => {
            engine.set_ghost_unselected(enabled);
        }
        EditorCommand::SetGhostOpacity(opacity) => {
            engine.set_ghost_opacity(opacity);
        }
        EditorCommand::SetFillStyle(fill) => {
            apply_to_selected_or_roots(engine, |graph, id| graph.set_fill_style(id, fill));
            session.dirty = true;
        }
        EditorCommand::SetEdgeStyle(edges) => {
            apply_to_selected_or_roots(engine, |graph, id| graph.set_edge_style(id, edges));
            session.dirty = true;
        }
        EditorCommand::SetFeatureEdgeColor(c) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_feature_edge_color(c);
            }
        }
        EditorCommand::SetWireframeEdgeColor(c) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_wireframe_edge_color(c);
            }
        }
        EditorCommand::SetHiddenEdgeColor(c) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_hidden_edge_color(c);
            }
        }
        EditorCommand::SetCreaseAngle(deg) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_feature_edge_crease_angle(deg);
            }
            engine.world.collector.invalidate_mesh_cache();
        }
        EditorCommand::SetSsEdgeThreshold(t) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_ss_edge_threshold(t);
            }
        }
        EditorCommand::SetDisplayMode(mode) => {
            engine.set_display_mode(display_mode_from_editor(mode));
        }
        EditorCommand::SetGridEnabled(enabled) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_grid_enabled(enabled);
            }
        }
        EditorCommand::SetHudEnabled(enabled) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_hud_enabled(enabled);
            }
        }
        EditorCommand::SetVsyncEnabled(enabled) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_vsync(enabled);
            }
        }
        EditorCommand::SetHdrPostProcessing(enabled) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_hdr_post_processing(enabled);
            }
        }
        EditorCommand::CycleIbl => {
            if let Some(r) = engine.renderer.as_mut() {
                r.cycle_ibl_preset();
            }
        }
        EditorCommand::SetIblPreset(preset) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_ibl_preset(preset);
            }
        }
        EditorCommand::SetOutlineWidth(w) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_outline_width(w);
            }
        }
        EditorCommand::SetOutlineColor(c) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_outline_color(c);
            }
        }
        EditorCommand::SetCadDisplayTier(tier) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_display_tier(tier);
            }
        }
        EditorCommand::SetAdaptiveQualityMode(mode) => {
            session.adaptive_quality_mode = mode;
            engine.set_adaptive_quality(adaptive_from_editor(mode));
        }
        EditorCommand::SetRenderFeature {
            feature_name,
            enabled,
        } => {
            if let Some(r) = engine.renderer.as_mut() {
                match feature_name {
                    "taa" => r.set_taa(enabled),
                    "motion_blur" => r.set_motion_blur(enabled),
                    "ssr" => r.set_ssr(enabled),
                    "color_grading" => r.set_color_grading(enabled),
                    "dof" => r.set_dof(enabled),
                    "volumetric_fog" => r.set_volumetric_fog(enabled),
                    "cluster_lights" => r.set_cluster_lights(enabled),
                    "omni_shadows" => r.set_omni_shadows(enabled),
                    "ldr_fxaa" => r.set_ldr_fxaa(enabled),
                    "screen_space_edges" => r.set_screen_space_edges(enabled),
                    "screen_space_selection_outline" => {
                        r.set_screen_space_selection_outline(enabled)
                    }
                    "gpu_cull" => r.set_gpu_culling(enabled),
                    "parallel_traversal" => r.set_parallel_traversal(enabled),
                    _ => {}
                }
            }
        }
        EditorCommand::SetGpuCulling(enabled) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_gpu_culling(enabled);
            }
        }
        EditorCommand::SetParallelTraversal(enabled) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_parallel_traversal(enabled);
            }
        }
        EditorCommand::SetCsmShadow {
            resolution,
            cascade_count,
        } => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_csm_shadow(resolution, cascade_count);
            }
        }
        EditorCommand::SetInteractionRenderScale(scale) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_interaction_render_scale(scale);
            }
        }
        EditorCommand::SetBgMode(mode) => {
            session.background.mode = mode;
            engine.set_background(session.background.clone());
        }
        EditorCommand::SetBgTopColor(c) => {
            session.background.top_color = c;
            session.background.clear_color = c;
            engine.set_background(session.background.clone());
        }
        EditorCommand::SetBgBotColor(c) => {
            session.background.bot_color = c;
            engine.set_background(session.background.clone());
        }
        EditorCommand::SetBgImage(path) => {
            session.background.mode = rc3d_render::background::BgMode::Image;
            session.background.image_path = Some(path.to_string_lossy().into_owned());
            engine.set_background(session.background.clone());
        }
        EditorCommand::SetPostEffects {
            vignette,
            chromatic,
            bloom,
            grain,
        } => {
            engine.set_post_effects(vignette, chromatic, bloom, grain);
        }
        EditorCommand::SetPostStylize { halftone, glitch } => {
            engine.set_post_stylize(halftone, glitch);
        }
        EditorCommand::CycleDisplayMode => {
            let next = match engine.renderer.as_ref().map(|r| r.display_mode()) {
                Some(rc3d_core::DisplayMode::Wireframe) => {
                    crate::ui::types::EditorDisplayMode::Shaded
                }
                Some(rc3d_core::DisplayMode::Shaded) => {
                    crate::ui::types::EditorDisplayMode::ShadedWithEdges
                }
                Some(rc3d_core::DisplayMode::ShadedWithEdges) => {
                    crate::ui::types::EditorDisplayMode::HiddenLine
                }
                Some(rc3d_core::DisplayMode::HiddenLine) => {
                    crate::ui::types::EditorDisplayMode::Flat
                }
                Some(rc3d_core::DisplayMode::Flat) => {
                    crate::ui::types::EditorDisplayMode::FlatWithEdge
                }
                _ => crate::ui::types::EditorDisplayMode::Wireframe,
            };
            engine.set_display_mode(display_mode_from_editor(next));
        }
        _ => {}
    }
}

fn apply_to_selected_or_roots(engine: &mut Engine, mut f: impl FnMut(&mut SceneGraph, NodeId)) {
    let selected: Vec<NodeId> = engine
        .world
        .graph
        .selected_nodes()
        .iter()
        .copied()
        .collect();
    let targets = if selected.is_empty() {
        engine.world.graph.roots().to_vec()
    } else {
        selected
    };
    for id in targets {
        f(&mut engine.world.graph, id);
    }
}

fn display_mode_from_editor(mode: EditorDisplayMode) -> DisplayMode {
    match mode {
        EditorDisplayMode::Wireframe => DisplayMode::Wireframe,
        EditorDisplayMode::Shaded => DisplayMode::Shaded,
        EditorDisplayMode::ShadedWithEdges => DisplayMode::ShadedWithEdges,
        EditorDisplayMode::HiddenLine => DisplayMode::HiddenLine,
        EditorDisplayMode::Flat => DisplayMode::Flat,
        EditorDisplayMode::FlatWithEdge => DisplayMode::FlatWithEdge,
    }
}

fn adaptive_from_editor(mode: AdaptiveQualityMode) -> AdaptiveControl {
    match mode {
        AdaptiveQualityMode::Off => AdaptiveControl::Disabled,
        AdaptiveQualityMode::On => AdaptiveControl::Dynamic {
            allow_downgrade: true,
        },
        AdaptiveQualityMode::AutoIdleLock => AdaptiveControl::Locked,
    }
}
