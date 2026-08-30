//! FreeCAD-style 26-face navigation cube (6 faces + 12 edges + 8 corners).

use std::sync::OnceLock;

use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};
use rc3d_core::math::{Mat4, Vec3};

use crate::commands::EditorCommand;
use crate::ui::i18n::t;
use crate::ui::types::{EditorChromeState, EditorUiContext};

const SIZE: f32 = 118.0;
const MARGIN: f32 = 10.0;
const CHAMFER: f32 = 0.32;
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
    if n.dot(center) < 0.0 {
        n = -n;
        let mut flipped = verts;
        flipped.reverse();
        out.push(NavFace {
            verts: flipped,
            normal: n,
            label,
        });
    } else {
        out.push(NavFace {
            verts,
            normal: n,
            label,
        });
    }
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

fn shade(base: Color32, n: Vec3, hover: bool) -> Color32 {
    let light = Vec3::new(0.35, 0.8, 0.45).normalize();
    let wrap = 0.5 + 0.5 * n.dot(light).clamp(-1.0, 1.0);
    let mut r = base.r() as f32 * wrap;
    let mut g = base.g() as f32 * wrap;
    let mut b = base.b() as f32 * wrap;
    if hover {
        r = r * 0.45 + 96.0;
        g = g * 0.45 + 170.0;
        b = b * 0.45 + 240.0;
    }
    Color32::from_rgb(r as u8, g as u8, b as u8)
}

pub(super) fn draw_nav_cube(
    ui: &mut Ui,
    scene_rect: Rect,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let rect = Rect::from_min_size(
        Pos2::new(scene_rect.min.x + MARGIN, scene_rect.min.y + MARGIN),
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
    let painter = ui.painter_at(rect);
    let pal = ui_ctx.ui_theme.palette();
    let loc = ui_ctx.ui_locale;
    let backdrop = if pal.dark {
        Color32::from_rgba_unmultiplied(18, 20, 24, 150)
    } else {
        Color32::from_rgba_unmultiplied(240, 240, 244, 180)
    };

    painter.circle_filled(origin, SIZE * 0.48, backdrop);

    struct Drawn {
        pts: Vec<Pos2>,
        depth: f32,
        normal: Vec3,
        label: Option<&'static str>,
        dir: [f32; 3],
        kind: u8,
    }

    let mut drawn = Vec::with_capacity(26);
    for face in faces() {
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
        let kind = if face.label.is_some() {
            0
        } else if face.verts.len() == 3 {
            2
        } else {
            1
        };
        drawn.push(Drawn {
            pts,
            depth,
            normal: face.normal,
            label: face.label,
            dir: face.normal.to_array(),
            kind,
        });
    }
    drawn.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal));

    let mut hit: Option<usize> = None;
    if let Some(pos) = pointer {
        for (i, f) in drawn.iter().enumerate().rev() {
            if point_in_poly(pos, &f.pts) {
                hit = Some(i);
                break;
            }
        }
    }

    let stroke = Stroke::new(
        1.0_f32,
        if pal.dark {
            Color32::from_rgb(28, 30, 36)
        } else {
            Color32::from_rgb(160, 166, 176)
        },
    );
    for (i, f) in drawn.iter().enumerate() {
        let hover = hit == Some(i);
        let base = match f.kind {
            0 => Color32::from_rgb(232, 234, 238),
            1 => Color32::from_rgb(196, 202, 214),
            _ => Color32::from_rgb(168, 178, 196),
        };
        painter.add(egui::Shape::convex_polygon(
            f.pts.clone(),
            shade(base, f.normal, hover),
            stroke,
        ));
        if let Some(key) = f.label {
            let c = f.pts.iter().copied().fold(Pos2::ZERO, |a, p| {
                Pos2::new(a.x + p.x, a.y + p.y)
            });
            let n = f.pts.len() as f32;
            let center = Pos2::new(c.x / n, c.y / n);
            let font = egui::FontId::proportional(10.0);
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                t(loc, key),
                font,
                Color32::from_rgb(32, 36, 44),
            );
        }
    }

    if response.clicked() {
        if let Some(i) = hit {
            push(EditorCommand::SetViewFromDirection(drawn[i].dir));
        }
    }

    response.on_hover_cursor(egui::CursorIcon::PointingHand);
}
