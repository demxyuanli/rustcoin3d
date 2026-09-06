use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_editor::{
    CaseKind, CaseListItem, CaseParamView, CaseStepView, EditorInteractionState, EditorSession,
};
use rc3d_engine_api::{CameraController, Engine};
use rc3d_scene::node_data::NodeData;
use rc3d_scene::SceneGraph;

use super::builders;
use super::common::{first_dir_light_mut, first_material_mut, first_section_mut};

#[derive(Clone, Copy, Debug)]
pub struct CaseParamDef {
    pub id: &'static str,
    pub label_key: &'static str,
    pub default: f32,
    pub min: f32,
    pub max: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct CaseStepDef {
    pub label_key: &'static str,
}

pub struct CaseDef {
    pub id: &'static str,
    pub title_key: &'static str,
    pub kind: CaseKind,
    pub category: &'static str,
    pub params: &'static [CaseParamDef],
    pub steps: &'static [CaseStepDef],
    pub build: fn() -> SceneGraph,
    pub apply_engine: fn(&mut Engine),
    pub apply_param: fn(&mut Engine, &str, f32),
    pub run_step: fn(&mut Engine, &mut EditorInteractionState, usize),
}

#[derive(Clone, Debug)]
pub struct ActiveCase {
    pub id: String,
    pub params: Vec<(String, f32)>,
    pub step: usize,
}

fn noop_param(_: &mut Engine, _: &str, _: f32) {}
fn noop_step(_: &mut Engine, _: &mut EditorInteractionState, _: usize) {}

fn shaded(engine: &mut Engine) {
    engine.set_display_mode(DisplayMode::Shaded);
}

fn apply_pbr_ball(engine: &mut Engine, id: &str, value: f32) {
    match id {
        "metallic" => {
            if let Some(m) = first_material_mut(engine.scene_mut()) {
                m.metallic = value;
            }
        }
        "roughness" => {
            if let Some(m) = first_material_mut(engine.scene_mut()) {
                m.roughness = value;
            }
        }
        "light" => {
            if let Some(l) = first_dir_light_mut(engine.scene_mut()) {
                l.intensity = value;
            }
        }
        _ => {}
    }
}

fn apply_shadows(engine: &mut Engine, id: &str, value: f32) {
    if id == "light" {
        if let Some(l) = first_dir_light_mut(engine.scene_mut()) {
            l.intensity = value;
        }
    }
}

fn apply_post(engine: &mut Engine, id: &str, value: f32) {
    let r = engine.renderer.as_ref();
    let mut vig = r.map(|r| r.post_fx_params.vignette).unwrap_or(0.3);
    let mut chr = r.map(|r| r.post_fx_params.chromatic).unwrap_or(0.0);
    let mut bloom = r.map(|r| r.post_fx_params.bloom_str).unwrap_or(0.8);
    let mut grain = r.map(|r| r.post_fx_params.grain).unwrap_or(0.0);
    let mut half = r.map(|r| r.post_fx_params.halftone).unwrap_or(0.0);
    let mut glitch = r.map(|r| r.post_fx_params.glitch).unwrap_or(0.0);
    match id {
        "vignette" => vig = value,
        "bloom" => bloom = value,
        "grain" => grain = value,
        "halftone" => half = value,
        "glitch" => glitch = value,
        "chromatic" => chr = value,
        _ => return,
    }
    engine.set_post_effects(vig, chr, bloom, grain);
    engine.set_post_stylize(half, glitch);
}

fn apply_wboit_engine(engine: &mut Engine) {
    shaded(engine);
    if let Some(r) = engine.renderer.as_mut() {
        r.enable_wboit = true;
    }
}

fn apply_volumetric_engine(engine: &mut Engine) {
    shaded(engine);
    if let Some(r) = engine.renderer.as_mut() {
        r.enable_volumetric_fog = true;
    }
}

fn apply_section(engine: &mut Engine, id: &str, value: f32) {
    if id == "height" {
        if let Some(s) = first_section_mut(engine.scene_mut()) {
            s.plane[3] = -value;
        }
    } else if id == "enabled" {
        if let Some(s) = first_section_mut(engine.scene_mut()) {
            s.enabled = value > 0.5;
        }
    }
}

fn apply_explode(engine: &mut Engine, id: &str, value: f32) {
    if id != "factor" {
        return;
    }
    let graph = engine.scene_mut();
    let ids: Vec<_> = graph
        .all_node_ids()
        .into_iter()
        .filter(|&nid| {
            graph
                .get(nid)
                .and_then(|e| e.name.as_deref())
                .is_some_and(|n| n.starts_with("part_"))
        })
        .collect();
    for (i, nid) in ids.into_iter().enumerate() {
        if let Some(children) = graph.children(nid).map(|c| c.to_vec()) {
            for c in children {
                if let Some(e) = graph.get_mut(c) {
                    if let NodeData::Transform(t) = &mut e.data {
                        t.translation.y = 1.0 + i as f32 + value * (i as f32 + 1.0) * 0.8;
                    }
                }
            }
        }
    }
}

fn step_orbit(engine: &mut Engine, _: &mut EditorInteractionState, step: usize) {
    let dir = if step % 2 == 0 { 0.15 } else { -0.15 };
    engine.orbit_view(dir, 0.05);
}

fn step_walk(engine: &mut Engine, _: &mut EditorInteractionState, step: usize) {
    let on = step == 0;
    if let Some(vc) = engine.viewport_cameras.active_mut() {
        vc.controller.walk_mode = on;
    }
    engine.controller.walk_mode = on;
}

fn step_section(engine: &mut Engine, _: &mut EditorInteractionState, step: usize) {
    if let Some(s) = first_section_mut(engine.scene_mut()) {
        s.enabled = step == 0;
    }
}

fn step_fit(engine: &mut Engine, _: &mut EditorInteractionState, _: usize) {
    let _ = engine;
    // Host may call FitAll via separate command; nudge camera distance.
    engine.controller.distance = (engine.controller.distance * 0.92).max(2.0);
}

macro_rules! params {
    ($(($id:literal, $key:literal, $def:expr, $min:expr, $max:expr)),* $(,)?) => {
        &[$(CaseParamDef { id: $id, label_key: $key, default: $def, min: $min, max: $max }),*]
    };
}

macro_rules! steps {
    ($($key:literal),* $(,)?) => {
        &[$(CaseStepDef { label_key: $key }),*]
    };
}

pub static ALL_CASES: &[CaseDef] = &[
    CaseDef {
        id: "hello",
        title_key: "case.hello",
        kind: CaseKind::Param,
        category: "basics",
        params: params!(("light", "case.param.light", 1.2, 0.1, 3.0)),
        steps: &[],
        build: builders::hello,
        apply_engine: shaded,
        apply_param: apply_shadows,
        run_step: noop_step,
    },
    CaseDef {
        id: "pbr-ball",
        title_key: "case.pbr_ball",
        kind: CaseKind::Param,
        category: "materials",
        params: params!(
            ("metallic", "case.param.metallic", 0.85, 0.0, 1.0),
            ("roughness", "case.param.roughness", 0.25, 0.05, 1.0),
            ("light", "case.param.light", 1.4, 0.1, 3.0),
        ),
        steps: &[],
        build: builders::pbr_ball,
        apply_engine: shaded,
        apply_param: apply_pbr_ball,
        run_step: noop_step,
    },
    CaseDef {
        id: "shadows",
        title_key: "case.shadows",
        kind: CaseKind::Param,
        category: "lighting",
        params: params!(("light", "case.param.light", 1.3, 0.1, 4.0)),
        steps: &[],
        build: builders::shadows,
        apply_engine: shaded,
        apply_param: apply_shadows,
        run_step: noop_step,
    },
    CaseDef {
        id: "pbr-grid",
        title_key: "case.pbr_grid",
        kind: CaseKind::Param,
        category: "materials",
        params: params!(("light", "case.param.light", 1.5, 0.1, 4.0)),
        steps: &[],
        build: builders::pbr_grid,
        apply_engine: shaded,
        apply_param: apply_shadows,
        run_step: noop_step,
    },
    CaseDef {
        id: "area-light",
        title_key: "case.area_light",
        kind: CaseKind::Param,
        category: "lighting",
        params: params!(("light", "case.param.light", 0.2, 0.0, 2.0)),
        steps: &[],
        build: builders::area_light,
        apply_engine: shaded,
        apply_param: apply_shadows,
        run_step: noop_step,
    },
    CaseDef {
        id: "post-fx",
        title_key: "case.post_fx",
        kind: CaseKind::Param,
        category: "post",
        params: params!(
            ("bloom", "case.param.bloom", 0.8, 0.0, 2.0),
            ("vignette", "case.param.vignette", 0.35, 0.0, 1.0),
            ("grain", "case.param.grain", 0.1, 0.0, 1.0),
            ("halftone", "case.param.halftone", 0.0, 0.0, 1.0),
            ("glitch", "case.param.glitch", 0.0, 0.0, 1.0),
        ),
        steps: &[],
        build: builders::post_fx,
        apply_engine: shaded,
        apply_param: apply_post,
        run_step: noop_step,
    },
    CaseDef {
        id: "volumetric",
        title_key: "case.volumetric",
        kind: CaseKind::Param,
        category: "post",
        params: params!(("light", "case.param.light", 1.3, 0.1, 4.0)),
        steps: &[],
        build: builders::volumetric,
        apply_engine: apply_volumetric_engine,
        apply_param: apply_shadows,
        run_step: noop_step,
    },
    CaseDef {
        id: "wboit",
        title_key: "case.wboit",
        kind: CaseKind::Param,
        category: "materials",
        params: params!(("light", "case.param.light", 1.2, 0.1, 3.0)),
        steps: &[],
        build: builders::wboit,
        apply_engine: apply_wboit_engine,
        apply_param: apply_shadows,
        run_step: noop_step,
    },
    CaseDef {
        id: "stereo",
        title_key: "case.stereo",
        kind: CaseKind::Param,
        category: "basics",
        params: params!(("light", "case.param.light", 1.2, 0.1, 3.0)),
        steps: &[],
        build: builders::stereo,
        apply_engine: shaded,
        apply_param: apply_shadows,
        run_step: noop_step,
    },
    CaseDef {
        id: "anim-spin",
        title_key: "case.anim_spin",
        kind: CaseKind::Process,
        category: "tools",
        params: &[],
        steps: steps!("case.step.orbit_left", "case.step.orbit_right"),
        build: builders::anim_spin,
        apply_engine: shaded,
        apply_param: noop_param,
        run_step: step_orbit,
    },
    CaseDef {
        id: "pick-demo",
        title_key: "case.pick_demo",
        kind: CaseKind::Process,
        category: "tools",
        params: &[],
        steps: steps!("case.step.fit"),
        build: builders::pick_demo,
        apply_engine: shaded,
        apply_param: noop_param,
        run_step: step_fit,
    },
    CaseDef {
        id: "section",
        title_key: "case.section",
        kind: CaseKind::Both,
        category: "tools",
        params: params!(
            ("height", "case.param.section_h", 0.9, 0.0, 2.5),
            ("enabled", "case.param.section_on", 1.0, 0.0, 1.0),
        ),
        steps: steps!("case.step.section_on", "case.step.section_off"),
        build: builders::section_demo,
        apply_engine: shaded,
        apply_param: apply_section,
        run_step: step_section,
    },
    CaseDef {
        id: "explode",
        title_key: "case.explode",
        kind: CaseKind::Both,
        category: "tools",
        params: params!(("factor", "case.param.explode", 0.0, 0.0, 2.0)),
        steps: &[],
        build: builders::explode_demo,
        apply_engine: shaded,
        apply_param: apply_explode,
        run_step: noop_step,
    },
    CaseDef {
        id: "walk",
        title_key: "case.walk",
        kind: CaseKind::Process,
        category: "tools",
        params: &[],
        steps: steps!("case.step.walk_on", "case.step.walk_off"),
        build: builders::walk_demo,
        apply_engine: shaded,
        apply_param: noop_param,
        run_step: step_walk,
    },
    CaseDef {
        id: "torus",
        title_key: "case.torus",
        kind: CaseKind::Param,
        category: "geometry",
        params: params!(("light", "case.param.light", 1.3, 0.1, 3.0)),
        steps: &[],
        build: builders::torus,
        apply_engine: shaded,
        apply_param: apply_shadows,
        run_step: noop_step,
    },
    CaseDef {
        id: "text3d",
        title_key: "case.text3d",
        kind: CaseKind::Param,
        category: "geometry",
        params: params!(("light", "case.param.light", 1.2, 0.1, 3.0)),
        steps: &[],
        build: builders::text3d_billboard,
        apply_engine: shaded,
        apply_param: apply_shadows,
        run_step: noop_step,
    },
    CaseDef {
        id: "instancing",
        title_key: "case.instancing",
        kind: CaseKind::Param,
        category: "geometry",
        params: params!(("light", "case.param.light", 1.3, 0.1, 3.0)),
        steps: &[],
        build: builders::instancing,
        apply_engine: shaded,
        apply_param: apply_shadows,
        run_step: noop_step,
    },
    CaseDef {
        id: "scene-graph",
        title_key: "case.scene_graph",
        kind: CaseKind::Both,
        category: "basics",
        params: params!(("light", "case.param.light", 1.2, 0.1, 3.0)),
        steps: steps!("case.step.orbit_left"),
        build: builders::scene_graph_demo,
        apply_engine: shaded,
        apply_param: apply_shadows,
        run_step: step_orbit,
    },
    CaseDef {
        id: "material-showcase",
        title_key: "case.material_showcase",
        kind: CaseKind::Param,
        category: "materials",
        params: params!(("light", "case.param.light", 1.4, 0.1, 4.0)),
        steps: &[],
        build: builders::material_showcase,
        apply_engine: shaded,
        apply_param: apply_shadows,
        run_step: noop_step,
    },
    CaseDef {
        id: "env-reflect",
        title_key: "case.env_reflect",
        kind: CaseKind::Param,
        category: "lighting",
        params: params!(
            ("metallic", "case.param.metallic", 0.85, 0.0, 1.0),
            ("roughness", "case.param.roughness", 0.25, 0.05, 1.0),
        ),
        steps: &[],
        build: builders::env_reflect,
        apply_engine: shaded,
        apply_param: apply_pbr_ball,
        run_step: noop_step,
    },
    CaseDef {
        id: "light-link",
        title_key: "case.light_link",
        kind: CaseKind::Both,
        category: "lighting",
        params: params!(("light", "case.param.light", 1.3, 0.1, 4.0)),
        steps: &[],
        build: builders::light_link,
        apply_engine: shaded,
        apply_param: apply_shadows,
        run_step: noop_step,
    },
    CaseDef {
        id: "annotation",
        title_key: "case.annotation",
        kind: CaseKind::Process,
        category: "tools",
        params: &[],
        steps: steps!("case.step.fit"),
        build: builders::annotation_demo,
        apply_engine: shaded,
        apply_param: noop_param,
        run_step: step_fit,
    },
    CaseDef {
        id: "lines",
        title_key: "case.lines",
        kind: CaseKind::Param,
        category: "geometry",
        params: &[],
        steps: &[],
        build: builders::lines_demo,
        apply_engine: shaded,
        apply_param: noop_param,
        run_step: noop_step,
    },
    CaseDef {
        id: "nurbs-approx",
        title_key: "case.nurbs",
        kind: CaseKind::Param,
        category: "geometry",
        params: params!(("light", "case.param.light", 1.3, 0.1, 3.0)),
        steps: &[],
        build: builders::nurbs_approx,
        apply_engine: shaded,
        apply_param: apply_shadows,
        run_step: noop_step,
    },
];

pub fn catalog_list() -> Vec<CaseListItem> {
    ALL_CASES
        .iter()
        .map(|c| CaseListItem {
            id: c.id.to_string(),
            title_key: c.title_key,
            kind: c.kind,
            category: c.category.to_string(),
        })
        .collect()
}

pub fn find_case(id: &str) -> Option<&'static CaseDef> {
    ALL_CASES.iter().find(|c| c.id == id)
}

pub fn load_case(
    id: &str,
    engine: &mut Engine,
    interaction: &mut EditorInteractionState,
    session: &mut EditorSession,
) -> Option<ActiveCase> {
    let case = find_case(id)?;
    let graph = (case.build)();
    engine.load_scene(graph);
    engine.controller = CameraController::new(Vec3::ZERO, 10.0);
    engine.hidden_nodes.clear();
    *interaction = EditorInteractionState::default();
    session.history.clear();
    session.document_path = None;
    session.dirty = false;
    rc3d_engine_api::sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
    (case.apply_engine)(engine);
    let params = case
        .params
        .iter()
        .map(|p| (p.id.to_string(), p.default))
        .collect::<Vec<_>>();
    for (pid, val) in &params {
        (case.apply_param)(engine, pid, *val);
    }
    Some(ActiveCase {
        id: case.id.to_string(),
        params,
        step: 0,
    })
}

pub fn set_case_param(active: &mut ActiveCase, engine: &mut Engine, param_id: &str, value: f32) {
    let Some(case) = find_case(&active.id) else {
        return;
    };
    if let Some(slot) = active.params.iter_mut().find(|(k, _)| k == param_id) {
        slot.1 = value;
    }
    (case.apply_param)(engine, param_id, value);
}

pub fn run_case_step(
    active: &mut ActiveCase,
    engine: &mut Engine,
    interaction: &mut EditorInteractionState,
    step: usize,
) {
    let Some(case) = find_case(&active.id) else {
        return;
    };
    active.step = step;
    (case.run_step)(engine, interaction, step);
}

pub fn param_views(active: &ActiveCase) -> Vec<CaseParamView> {
    let Some(case) = find_case(&active.id) else {
        return Vec::new();
    };
    case.params
        .iter()
        .map(|p| {
            let value = active
                .params
                .iter()
                .find(|(k, _)| k == p.id)
                .map(|(_, v)| *v)
                .unwrap_or(p.default);
            CaseParamView {
                id: p.id.to_string(),
                label_key: p.label_key,
                value,
                min: p.min,
                max: p.max,
            }
        })
        .collect()
}

pub fn step_views(active: &ActiveCase) -> Vec<CaseStepView> {
    let Some(case) = find_case(&active.id) else {
        return Vec::new();
    };
    case.steps
        .iter()
        .enumerate()
        .map(|(i, s)| CaseStepView {
            index: i,
            label_key: s.label_key,
            active: i == active.step,
        })
        .collect()
}
