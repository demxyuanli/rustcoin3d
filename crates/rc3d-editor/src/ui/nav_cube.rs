//! FreeCAD-style 26-face navigation cube (6 faces + 12 edges + 8 corners).

use std::sync::OnceLock;

use egui::{Pos2, Rect, Sense, Ui, Vec2};
use rc3d_core::math::{Mat4, Vec3};

use crate::commands::EditorCommand;
use crate::ui::i18n::t;
use crate::ui::types::{EditorChromeState, EditorUiContext};
use rc3d_engine_api::{OverlayViewport, OVERLAY_NAV_CUBE};
use rc3d_scene::node_data::{
    Coordinate3Node, DirectionalLightNode, FaceMaterialGroup, HemisphereLightNode,
    IndexedFaceSetNode, MaterialNode, NodeData, SeparatorNode, Text3Node, TransformNode,
};
use rc3d_scene::SceneGraph;

pub const SIZE: f32 = 118.0;
pub const MARGIN: f32 = 10.0;
const CHAMFER: f32 = 0.45;
const ORBIT_SCALE: f32 = 0.012;

struct NavFace {
    verts: Vec<Vec3>,
    normal: Vec3,
    label: Option<&'static str>,
}

fn faces() -> &'static [NavFace] {
    static FACES: OnceLock<Vec<NavFace>> = OnceLock::new();
    FACES.get_or_init(build_faces)
}

fn build_faces() -> Vec<NavFace> {
    let t = 1.0 - CHAMFER;
    let mut out = Vec::with_capacity(26);

    let face_spec = [
        (2usize, 1.0f32, Some("nav.front")),
        (2, -1.0, Some("nav.back")),
        (1, 1.0, Some("nav.top")),
        (1, -1.0, Some("nav.bottom")),
        (0, 1.0, Some("nav.right")),
        (0, -1.0, Some("nav.left")),
    ];
    for &(axis, sign, label) in &face_spec {
        let (u, v) = match axis {
            0 => (1, 2),
            1 => (2, 0),
            _ => (0, 1),
        };
        let corners = [(-t, -t), (t, -t), (t, t), (-t, t)];
        let mut verts = Vec::with_capacity(4);
        for &(cu, cv) in &corners {
            let mut p = [0.0f32; 3];
            p[axis] = sign;
            p[u] = cu;
            p[v] = cv;
            verts.push(Vec3::from_array(p));
        }
        if sign < 0.0 {
            verts.reverse();
        }
        push_face(&mut out, verts, label);
    }

    for a in 0..3 {
        for b in (a + 1)..3 {
            for &sa in &[-1.0f32, 1.0] {
                for &sb in &[-1.0f32, 1.0] {
                    let k = 3 - a - b;
                    let mk = |ca: f32, cb: f32, ck: f32| {
                        let mut p = [0.0f32; 3];
                        p[a] = ca;
                        p[b] = cb;
                        p[k] = ck;
                        Vec3::from_array(p)
                    };
                    let verts = vec![
                        mk(sa, sb * t, -t),
                        mk(sa, sb * t, t),
                        mk(sa * t, sb, t),
                        mk(sa * t, sb, -t),
                    ];
                    push_face(&mut out, verts, None);
                }
            }
        }
    }

    for &sx in &[-1.0f32, 1.0] {
        for &sy in &[-1.0f32, 1.0] {
            for &sz in &[-1.0f32, 1.0] {
                let verts = vec![
                    Vec3::new(sx * t, sy * t, sz),
                    Vec3::new(sx * t, sy, sz * t),
                    Vec3::new(sx, sy * t, sz * t),
                ];
                push_face(&mut out, verts, None);
            }
        }
    }

    debug_assert_eq!(out.len(), 26);
    out
}

fn push_face(out: &mut Vec<NavFace>, verts: Vec<Vec3>, label: Option<&'static str>) {
    if verts.len() < 3 {
        return;
    }
    let mut n = (verts[1] - verts[0]).cross(verts[2] - verts[0]);
    if n.length_squared() < 1.0e-10 {
        return;
    }
    n = n.normalize();
    let center: Vec3 = verts.iter().copied().sum::<Vec3>() / verts.len() as f32;
    // Convex cube at the origin: the centroid direction is the outward face normal.
    let outward = center.normalize_or_zero();
    if outward.length_squared() < 1.0e-10 {
        return;
    }
    let mut verts = verts;
    if n.dot(outward) < 0.0 {
        verts.reverse();
    }
    out.push(NavFace {
        verts,
        normal: outward,
        label,
    });
}

fn view_matrix(from: Vec3, up: Vec3) -> Mat4 {
    let eye = from.normalize_or_zero();
    if eye.length_squared() < 1.0e-8 {
        return Mat4::look_at_rh(Vec3::Z, Vec3::ZERO, Vec3::Y);
    }
    let mut up = up.normalize_or_zero();
    if up.length_squared() < 1.0e-8 || eye.dot(up).abs() > 0.999 {
        up = if eye.y.abs() > 0.5 { -Vec3::Z } else { Vec3::Y };
    }
    Mat4::look_at_rh(eye, Vec3::ZERO, up)
}

fn project(v: Vec3, view: Mat4, origin: Pos2, half: f32) -> (Pos2, f32) {
    let p = view.transform_point3(v);
    (origin + Vec2::new(p.x * half, -p.y * half), p.z)
}

fn point_in_poly(p: Pos2, verts: &[Pos2]) -> bool {
    if verts.len() < 3 {
        return false;
    }
    let mut sign = 0i32;
    for i in 0..verts.len() {
        let a = verts[i];
        let b = verts[(i + 1) % verts.len()];
        let z = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        let s = if z > 1.0e-4 {
            1
        } else if z < -1.0e-4 {
            -1
        } else {
            0
        };
        if s == 0 {
            continue;
        }
        if sign == 0 {
            sign = s;
        } else if s != sign {
            return false;
        }
    }
    true
}

pub(super) fn draw_nav_cube(
    ui: &mut Ui,
    scene_rect: Rect,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let rect = Rect::from_min_size(
        Pos2::new(scene_rect.max.x - MARGIN - SIZE, scene_rect.min.y + MARGIN),
        Vec2::splat(SIZE),
    );
    chrome.nav_cube_rect_points = Some([rect.min.x, rect.min.y, rect.width(), rect.height()]);

    let response = ui.allocate_rect(rect, Sense::click_and_drag());
    chrome.nav_cube_dragging = response.dragged();

    if response.dragged() {
        let d = response.drag_delta();
        if d.length_sq() > 0.0 {
            push(EditorCommand::OrbitView {
                dx: d.x * ORBIT_SCALE,
                dy: d.y * ORBIT_SCALE,
            });
        }
    }

    let from = Vec3::from_array(ui_ctx.camera_from);
    let up = Vec3::from_array(ui_ctx.camera_up);
    let view = view_matrix(from, up);
    let origin = rect.center();
    let half = SIZE * 0.36;

    let pointer = response.hover_pos();

    struct Drawn {
        pts: Vec<Pos2>,
        depth: f32,
        dir: [f32; 3],
        slot: u32,
    }

    let mut drawn = Vec::with_capacity(26);
    for (slot, face) in faces().iter().enumerate() {
        if face.normal.dot(from.normalize_or_zero()) <= 0.04 {
            continue;
        }
        let mut pts = Vec::with_capacity(face.verts.len());
        let mut depth = 0.0f32;
        for &v in &face.verts {
            let (p, z) = project(v, view, origin, half);
            pts.push(p);
            depth += z;
        }
        depth /= face.verts.len() as f32;
        drawn.push(Drawn {
            pts,
            depth,
            dir: face.normal.to_array(),
            slot: slot as u32,
        });
    }
    drawn.sort_by(|a, b| {
        a.depth
            .partial_cmp(&b.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut hit: Option<usize> = None;
    if let Some(pos) = pointer {
        for (i, f) in drawn.iter().enumerate().rev() {
            if point_in_poly(pos, &f.pts) {
                hit = Some(i);
                break;
            }
        }
    }
    chrome.nav_cube_hover_slot = hit.map(|i| drawn[i].slot);

    if response.clicked() {
        if let Some(i) = hit {
            push(EditorCommand::SetViewFromDirection(drawn[i].dir));
        }
    }

    response.on_hover_cursor(egui::CursorIcon::PointingHand);
}

fn face_kind_color(kind: u8) -> Vec3 {
    match kind {
        0 => Vec3::new(0.91, 0.92, 0.94),
        1 => Vec3::new(0.77, 0.79, 0.84),
        _ => Vec3::new(0.66, 0.70, 0.77),
    }
}

fn face_kind(face: &NavFace) -> u8 {
    if face.label.is_some() {
        0
    } else if face.verts.len() == 3 {
        2
    } else {
        1
    }
}

fn face_material(color: Vec3, hover: bool) -> MaterialNode {
    let mut mat = MaterialNode::from_diffuse(color);
    mat.roughness = 0.55;
    mat.metallic = 0.0;
    if hover {
        mat.base_color = color * 0.5 + Vec3::new(0.22, 0.48, 0.92) * 0.5;
        mat.diffuse_color = mat.base_color;
        mat.emissive_color = Vec3::new(0.18, 0.38, 0.72);
    } else {
        mat.emissive_color = color * 0.06;
    }
    mat
}

fn face_basis(n: Vec3) -> (Vec3, Vec3) {
    let n = n.normalize_or_zero();
    let mut bit = Vec3::Y - n * n.dot(Vec3::Y);
    if bit.length_squared() < 0.04 {
        bit = -Vec3::Z - n * n.dot(-Vec3::Z);
    }
    let bit = bit.normalize_or_zero();
    let tan = bit.cross(n).normalize_or_zero();
    (tan, bit)
}

fn nav_label_key(name: &str) -> Option<&'static str> {
    match name {
        "nav.front" => Some("nav.front"),
        "nav.back" => Some("nav.back"),
        "nav.top" => Some("nav.top"),
        "nav.bottom" => Some("nav.bottom"),
        "nav.left" => Some("nav.left"),
        "nav.right" => Some("nav.right"),
        _ => None,
    }
}

fn visit_nodes(graph: &mut SceneGraph, mut visit: impl FnMut(&mut rc3d_scene::NodeEntry)) {
    let mut stack: Vec<rc3d_core::NodeId> = graph.roots().to_vec();
    while let Some(id) = stack.pop() {
        let kids = graph
            .get(id)
            .map(|e| e.children.clone())
            .unwrap_or_default();
        if let Some(e) = graph.get_mut(id) {
            visit(e);
        }
        stack.extend(kids);
    }
}

/// Update overlay labels / hover materials from editor chrome (call before GPU render).
pub fn sync_overlay(
    ov: &mut OverlayViewport,
    locale: crate::ui::i18n::UiLocale,
    theme: crate::ui::theme::UiTheme,
    hover_slot: Option<u32>,
) {
    let dark = theme.palette().dark;
    let label_color = if dark {
        [0.12, 0.14, 0.18, 1.0]
    } else {
        [0.16, 0.18, 0.22, 1.0]
    };
    visit_nodes(&mut ov.world.graph, |entry| match &mut entry.data {
        NodeData::IndexedFaceSet(ifs) => {
            for (i, mat) in ifs.materials.iter_mut().enumerate() {
                let kind = if i < 6 {
                    0
                } else if i < 18 {
                    1
                } else {
                    2
                };
                *mat = face_material(face_kind_color(kind), hover_slot == Some(i as u32));
            }
        }
        NodeData::Text3(text) => {
            if let Some(key) = entry.name.as_deref().and_then(nav_label_key) {
                text.string = t(locale, key).to_string();
                text.color = label_color;
            }
        }
        _ => {}
    });
}

/// Independent overlay world for the navigation cube (own scene graph + camera).
pub fn nav_cube_overlay() -> OverlayViewport {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    let light_id = graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(0.0, 0.0, -1.0),
            color: Vec3::ONE,
            intensity: 1.15,
            light_group: None,
        }),
    );
    graph.add_child(
        root,
        NodeData::HemisphereLight(HemisphereLightNode {
            sky_color: Vec3::new(0.92, 0.94, 0.98),
            ground_color: Vec3::new(0.42, 0.44, 0.48),
            intensity: 0.55,
            direction: Vec3::Y,
        }),
    );

    let mut points = Vec::new();
    let mut coord_index = Vec::new();
    let mut tri_slots = Vec::new();
    let mut materials = Vec::with_capacity(26);
    for (slot, face) in faces().iter().enumerate() {
        let base = points.len() as i32;
        points.extend_from_slice(&face.verts);
        let n = face.verts.len();
        for i in 0..n {
            coord_index.push(base + i as i32);
        }
        coord_index.push(-1);
        let kind = face_kind(face);
        materials.push(face_material(face_kind_color(kind), false));
        let tri_count = n.saturating_sub(2);
        for _ in 0..tri_count {
            tri_slots.push(slot as u32);
        }
    }

    graph.add_child(
        root,
        NodeData::Coordinate3(Coordinate3Node::from_points(points)),
    );
    graph.add_child(
        root,
        NodeData::IndexedFaceSet(IndexedFaceSetNode {
            coord_index,
            material_groups: FaceMaterialGroup::compact_from_triangle_slots(&tri_slots),
            materials,
            face_ids: Vec::new(),
        }),
    );

    for face in faces() {
        let Some(key) = face.label else {
            continue;
        };
        let center: Vec3 = face.verts.iter().copied().sum::<Vec3>() / face.verts.len() as f32;
        let (tan, bit) = face_basis(face.normal);
        let rotation = Mat4::from_cols(
            tan.extend(0.0),
            bit.extend(0.0),
            face.normal.extend(0.0),
            Vec3::ZERO.extend(1.0),
        );
        let isol = graph.add_child(root, NodeData::Separator(SeparatorNode));
        let tf = graph.add_child(
            isol,
            NodeData::Transform(TransformNode::from_trs(
                center + face.normal * 0.045,
                rotation,
                Vec3::ONE,
            )),
        );
        let text_id = graph.add_child(
            tf,
            NodeData::Text3(Text3Node {
                string: String::new(),
                position: Vec3::ZERO,
                size: 11.0,
                color: [0.12, 0.14, 0.18, 1.0],
                plane_aligned: true,
            }),
        );
        if let Some(e) = graph.get_mut(text_id) {
            e.name = Some(key.to_string());
        }
    }

    let mut ov = OverlayViewport::new(OVERLAY_NAV_CUBE, graph);
    ov.light_id = Some(light_id);
    ov.follow_main_camera = true;
    ov.orthographic = true;
    ov.ortho_half = 1.55;
    ov.eye_distance = 4.0;
    ov
}
