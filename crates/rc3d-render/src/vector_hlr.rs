//! Vector hardcopy of Fast Hidden Line edges (SVG).
//!
//! Projects crease / wireframe edges and classifies hidden vs visible by
//! ray-vs-triangle tests against filled draws. This is not HOOPS HIO analytic HLR.

use crate::render_action::DrawCall;
use glam::{Mat4, Vec3, Vec4};

const MAX_TRIS: usize = 24_000;
const MAX_EDGES: usize = 16_000;

/// Build an SVG of visible (solid) and hidden (dashed) feature edges.
pub fn hidden_line_svg(
    draw_calls: &[DrawCall],
    view: Mat4,
    proj: Mat4,
    camera_pos: Vec3,
    width: u32,
    height: u32,
) -> String {
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    let vp = proj * view;

    let tris = collect_world_triangles(draw_calls);
    let edges = collect_world_edges(draw_calls);

    let mut visible = String::new();
    let mut hidden = String::new();
    for (a, b) in edges {
        let Some((x0, y0)) = project_svg(vp, a, w, h) else {
            continue;
        };
        let Some((x1, y1)) = project_svg(vp, b, w, h) else {
            continue;
        };
        let mid = (a + b) * 0.5;
        let to_mid = mid - camera_pos;
        let dist = to_mid.length();
        if dist < 1e-5 {
            continue;
        }
        let dir = to_mid / dist;
        let occluded = edge_occluded(camera_pos, dir, dist, a, b, &tris);
        let line = format!(
            "    <line x1=\"{x0:.2}\" y1=\"{y0:.2}\" x2=\"{x1:.2}\" y2=\"{y1:.2}\" />\n"
        );
        if occluded {
            hidden.push_str(&line);
        } else {
            visible.push_str(&line);
        }
    }

    let mut out = String::with_capacity(visible.len() + hidden.len() + 256);
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w:.0}\" height=\"{h:.0}\" viewBox=\"0 0 {w:.0} {h:.0}\">\n"
    ));
    out.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"#f4f4f2\" />\n");
    out.push_str("  <g stroke=\"#8a8a88\" fill=\"none\" stroke-width=\"0.75\" stroke-dasharray=\"4 3\">\n");
    out.push_str(&hidden);
    out.push_str("  </g>\n");
    out.push_str("  <g stroke=\"#1a1a1a\" fill=\"none\" stroke-width=\"1.1\">\n");
    out.push_str(&visible);
    out.push_str("  </g>\n</svg>\n");
    out
}

fn collect_world_triangles(draw_calls: &[DrawCall]) -> Vec<[Vec3; 3]> {
    let mut tris = Vec::new();
    for dc in draw_calls {
        if !dc.appearance().wants_filled() {
            continue;
        }
        let model = dc.model_matrix;
        if let Some(indices) = dc.indices.as_ref() {
            let start = dc.index_first as usize;
            let count = if dc.index_draw_count == 0 {
                indices.len().saturating_sub(start)
            } else {
                dc.index_draw_count as usize
            };
            let end = (start + count).min(indices.len());
            let slice = &indices[start..end];
            for tri in slice.chunks_exact(3) {
                if let Some(t) = world_tri(dc, model, tri[0], tri[1], tri[2]) {
                    tris.push(t);
                    if tris.len() >= MAX_TRIS {
                        return tris;
                    }
                }
            }
        } else {
            let n = dc.vertices.len();
            let mut i = 0;
            while i + 2 < n {
                if let Some(t) = world_tri(dc, model, i as u32, (i + 1) as u32, (i + 2) as u32) {
                    tris.push(t);
                    if tris.len() >= MAX_TRIS {
                        return tris;
                    }
                }
                i += 3;
            }
        }
    }
    tris
}

fn world_tri(dc: &DrawCall, model: Mat4, i0: u32, i1: u32, i2: u32) -> Option<[Vec3; 3]> {
    let v0 = dc.vertices.get(i0 as usize)?;
    let v1 = dc.vertices.get(i1 as usize)?;
    let v2 = dc.vertices.get(i2 as usize)?;
    Some([
        (model * Vec4::from((Vec3::from_array(v0.position), 1.0))).truncate(),
        (model * Vec4::from((Vec3::from_array(v1.position), 1.0))).truncate(),
        (model * Vec4::from((Vec3::from_array(v2.position), 1.0))).truncate(),
    ])
}

fn collect_world_edges(draw_calls: &[DrawCall]) -> Vec<(Vec3, Vec3)> {
    let mut edges = Vec::new();
    for dc in draw_calls {
        let src = if !dc.edge_positions.is_empty() {
            dc.edge_positions.as_slice()
        } else {
            dc.wireframe_edge_positions.as_slice()
        };
        let model = dc.model_matrix;
        for pair in src.chunks_exact(2) {
            let a = (model * Vec4::from((Vec3::from_array(pair[0]), 1.0))).truncate();
            let b = (model * Vec4::from((Vec3::from_array(pair[1]), 1.0))).truncate();
            edges.push((a, b));
            if edges.len() >= MAX_EDGES {
                return edges;
            }
        }
    }
    edges
}

fn project_svg(vp: Mat4, p: Vec3, w: f32, h: f32) -> Option<(f32, f32)> {
    let clip = vp * Vec4::from((p, 1.0));
    if clip.w <= 1e-5 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.x < -1.2 || ndc.x > 1.2 || ndc.y < -1.2 || ndc.y > 1.2 {
        return None;
    }
    let x = (ndc.x * 0.5 + 0.5) * w;
    let y = (1.0 - (ndc.y * 0.5 + 0.5)) * h;
    Some((x, y))
}

fn edge_occluded(
    origin: Vec3,
    dir: Vec3,
    dist: f32,
    a: Vec3,
    b: Vec3,
    tris: &[[Vec3; 3]],
) -> bool {
    for t in tris {
        if shares_edge(t, a, b) {
            continue;
        }
        if let Some(hit) = ray_triangle(origin, dir, t[0], t[1], t[2]) {
            if hit < dist - 1e-3 {
                return true;
            }
        }
    }
    false
}

fn shares_edge(tri: &[Vec3; 3], a: Vec3, b: Vec3) -> bool {
    let mut hit_a = false;
    let mut hit_b = false;
    for &v in tri {
        if (v - a).length_squared() < 1e-8 {
            hit_a = true;
        }
        if (v - b).length_squared() < 1e-8 {
            hit_b = true;
        }
    }
    hit_a && hit_b
}

fn ray_triangle(orig: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let pvec = dir.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = orig - v0;
    let u = tvec.dot(pvec) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(e1);
    let v = dir.dot(qvec) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(qvec) * inv;
    if t > 1e-4 {
        Some(t)
    } else {
        None
    }
}
