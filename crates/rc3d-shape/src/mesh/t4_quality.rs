//! T4 mesh quality metrics — surface-to-mesh deflection and optional reference comparison.
//! OCC parity: 95th percentile surface sample distance ≤ 2× linear deflection.

use rc3d_core::math::Vec3;

use super::face_uv::point_in_trim;
use super::BRepMeshConfig;
use crate::geom::SurfaceGeom;
use crate::store::BRepStore;
use crate::topo::{FaceKey, Orientation, ShellKey, WireKey};
use crate::mesh_result::MeshResult;

#[derive(Debug, Clone, Default)]
pub struct DeflectionMetrics {
    pub sample_count: usize,
    pub max: f32,
    pub p95: f32,
}

#[derive(Debug, Clone, Default)]
pub struct HausdorffMetrics {
    pub sample_count: usize,
    pub max_a_to_b: f32,
    pub p95_a_to_b: f32,
    pub max_b_to_a: f32,
    pub p95_b_to_a: f32,
    pub symmetric_max: f32,
    pub symmetric_p95: f32,
}

/// Geometric deflection: mesh triangle centroids → nearest trimmed face surface.
pub fn measure_shell_deflection(
    reg: &BRepStore,
    shell_key: ShellKey,
    mesh: &MeshResult,
    _grid: u32,
) -> DeflectionMetrics {
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return DeflectionMetrics::default(),
    };
    let mut dists = Vec::new();
    sample_mesh_to_surface(reg, shell, mesh, &mut dists);
    dists_to_metrics(&dists)
}

/// Per-triangle centroid distance to the nearest face surface in the shell.
fn sample_mesh_to_surface(
    reg: &BRepStore,
    shell: &crate::topo::BRepShell,
    mesh: &MeshResult,
    out: &mut Vec<f32>,
) {
    for chunk in mesh.indices.chunks(4) {
        if chunk.len() < 3 {
            continue;
        }
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;
        if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
            continue;
        }
        let c = (mesh.vertices[i0] + mesh.vertices[i1] + mesh.vertices[i2]) * (1.0 / 3.0);
        let d = closest_surface_distance(reg, shell, c);
        if d.is_finite() && d < f32::MAX {
            out.push(d);
        }
    }
}

fn closest_surface_distance(reg: &BRepStore, shell: &crate::topo::BRepShell, point: Vec3) -> f32 {
    let mut best = f32::MAX;
    for &(face_key, _) in &shell.faces {
        let Some(face) = reg.faces.get(face_key) else {
            continue;
        };
        let inv_tol = face.tolerance.max(1e-3);
        let Some(uv) = face
            .surface
            .project(point)
            .or_else(|| face.surface.inverse_native_uv(point, inv_tol))
        else {
            continue;
        };
        let outer_uv = wire_uv_polygon(reg, face_key, face.outer_wire, &face.surface, inv_tol);
        if outer_uv.len() >= 3 {
            let inner_uv: Vec<Vec<(f32, f32)>> = face
                .inner_wires
                .iter()
                .map(|&wk| wire_uv_polygon(reg, face_key, wk, &face.surface, inv_tol))
                .filter(|p| p.len() >= 3)
                .collect();
            if !point_in_trim(uv.0, uv.1, &outer_uv, &inner_uv) {
                continue;
            }
        }
        let on = face.surface.d0_native(uv.0, uv.1);
        best = best.min((point - on).length());
    }
    if best == f32::MAX {
        0.0
    } else {
        best
    }
}

fn wire_uv_polygon(
    reg: &BRepStore,
    face_key: FaceKey,
    wire_key: WireKey,
    surface: &SurfaceGeom,
    inv_tol: f32,
) -> Vec<(f32, f32)> {
    let wire = match reg.wires.get(wire_key) {
        Some(w) => w,
        None => return Vec::new(),
    };
    let mut poly = Vec::new();
    const STEPS: u32 = 4;
    for &(ek, orient) in &wire.edges {
        let Some(edge) = reg.edges.get(ek) else {
            continue;
        };
        if let Some(pcurve) = edge.pcurves.get(&face_key) {
            for i in 0..=STEPS {
                let t = i as f32 / STEPS as f32;
                let t = if orient == Orientation::Reversed {
                    1.0 - t
                } else {
                    t
                };
                let uv = pcurve.d0(t);
                push_uv(&mut poly, (uv.x, uv.y));
            }
        } else {
            let verts = if orient == Orientation::Reversed {
                [edge.v_high, edge.v_low]
            } else {
                [edge.v_low, edge.v_high]
            };
            for vk in verts {
                if let Some(v) = reg.vertices.get(vk) {
                    if let Some(uv) = surface
                        .project(v.position)
                        .or_else(|| surface.inverse_native_uv(v.position, inv_tol))
                    {
                        push_uv(&mut poly, uv);
                    }
                }
            }
        }
    }
    poly
}

fn push_uv(poly: &mut Vec<(f32, f32)>, uv: (f32, f32)) {
    if poly.last().copied() != Some(uv) {
        poly.push(uv);
    }
}

fn dists_to_metrics(dists: &[f32]) -> DeflectionMetrics {
    if dists.is_empty() {
        return DeflectionMetrics::default();
    }
    let mut sorted = dists.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    DeflectionMetrics {
        sample_count: n,
        max: *sorted.last().unwrap_or(&0.0),
        p95: sorted[((n as f32 * 0.95) as usize).min(n.saturating_sub(1))],
    }
}

/// Closest distance from point to triangle soup.
pub fn point_to_mesh_distance(point: Vec3, mesh: &MeshResult) -> f32 {
    let mut best = f32::MAX;
    for chunk in mesh.indices.chunks(4) {
        if chunk.len() < 3 {
            continue;
        }
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;
        if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
            continue;
        }
        let d = closest_point_on_triangle(
            point,
            mesh.vertices[i0],
            mesh.vertices[i1],
            mesh.vertices[i2],
        );
        if d < best {
            best = d;
        }
    }
    if best == f32::MAX {
        0.0
    } else {
        best
    }
}

fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (p - a).length();
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (p - b).length();
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length();
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (p - c).length();
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length();
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length();
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (p - (a + ab * v + ac * w)).length()
}

/// One-sided Hausdorff: sample `a` triangle centroids → distance to `b`.
pub fn hausdorff_meshes(a: &MeshResult, b: &MeshResult, max_samples: usize) -> HausdorffMetrics {
    let a_to_b = sample_mesh_distances(a, b, max_samples);
    let b_to_a = sample_mesh_distances(b, a, max_samples);
    let p95 = |d: &[f32]| -> f32 {
        if d.is_empty() {
            return 0.0;
        }
        let mut s = d.to_vec();
        s.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let n = s.len();
        s[((n as f32 * 0.95) as usize).min(n.saturating_sub(1))]
    };
    let max = |d: &[f32]| d.iter().copied().fold(0.0f32, f32::max);
    HausdorffMetrics {
        sample_count: a_to_b.len().max(b_to_a.len()),
        max_a_to_b: max(&a_to_b),
        p95_a_to_b: p95(&a_to_b),
        max_b_to_a: max(&b_to_a),
        p95_b_to_a: p95(&b_to_a),
        symmetric_max: max(&a_to_b).max(max(&b_to_a)),
        symmetric_p95: p95(&a_to_b).max(p95(&b_to_a)),
    }
}

fn sample_mesh_distances(from: &MeshResult, to: &MeshResult, max_samples: usize) -> Vec<f32> {
    let tri_count = from.indices.len() / 4;
    if tri_count == 0 {
        return Vec::new();
    }
    let step = (tri_count / max_samples.max(1)).max(1);
    let mut out = Vec::new();
    for (ti, chunk) in from.indices.chunks(4).enumerate() {
        if ti % step != 0 || chunk.len() < 3 {
            continue;
        }
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;
        if i0 >= from.vertices.len() || i1 >= from.vertices.len() || i2 >= from.vertices.len() {
            continue;
        }
        let c = (from.vertices[i0] + from.vertices[i1] + from.vertices[i2]) * (1.0 / 3.0);
        out.push(point_to_mesh_distance(c, to));
    }
    out
}

/// T4 band: 95th percentile ≤ `2 × deflection_interior`.
pub fn deflection_within_band(metrics: &DeflectionMetrics, config: &BRepMeshConfig) -> bool {
    let band = config.face.deflection_interior * 2.0;
    metrics.p95 <= band
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_on_triangle_has_zero_distance() {
        let mesh = MeshResult {
            vertices: vec![
                Vec3::ZERO,
                Vec3::X,
                Vec3::Y,
            ],
            normals: vec![],
            indices: vec![0, 1, 2, -1],
        };
        let d = point_to_mesh_distance(Vec3::new(0.25, 0.25, 0.0), &mesh);
        assert!(d < 1e-4);
    }

    #[test]
    fn hausdorff_identical_meshes_near_zero() {
        let mesh = MeshResult {
            vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            normals: vec![],
            indices: vec![0, 1, 2, -1],
        };
        let h = hausdorff_meshes(&mesh, &mesh, 8);
        assert!(h.symmetric_p95 < 1e-4);
    }
}
