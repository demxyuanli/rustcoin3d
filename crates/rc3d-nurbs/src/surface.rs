use glam::Vec3;
use crate::basis::bspline_basis;
use crate::knot::{find_span, open_uniform_knots};
use crate::tessellate::tessellate_surface_adaptive;

/// A NURBS surface defined by a grid of homogeneous control points.
#[derive(Clone, Debug)]
pub struct NurbsSurface {
    pub control_points: Vec<Vec<[f32; 4]>>, // [u_count][v_count] = (wx, wy, wz, w)
    pub u_knots: Vec<f32>,
    pub v_knots: Vec<f32>,
    pub u_degree: usize,
    pub v_degree: usize,
}

impl NurbsSurface {
    /// Create a NURBS surface from a u×v grid of homogeneous control points.
    pub fn new(
        control_points: Vec<Vec<[f32; 4]>>,
        u_degree: usize,
        v_degree: usize,
    ) -> Self {
        let u_count = control_points.len();
        let v_count = if u_count > 0 {
            control_points[0].len()
        } else {
            0
        };
        let u_knots = open_uniform_knots(u_degree, u_count);
        let v_knots = open_uniform_knots(v_degree, v_count);
        Self { control_points, u_knots, v_knots, u_degree, v_degree }
    }

    /// Create a NURBS surface from non-rational control points (Vec3 grid).
    pub fn from_points_grid(
        points: &[Vec<Vec3>],
        u_degree: usize,
        v_degree: usize,
    ) -> Self {
        let cp: Vec<Vec<[f32; 4]>> = points
            .iter()
            .map(|row| row.iter().map(|p| [p.x, p.y, p.z, 1.0]).collect())
            .collect();
        Self::new(cp, u_degree, v_degree)
    }

    pub fn u_count(&self) -> usize {
        self.control_points.len()
    }

    pub fn v_count(&self) -> usize {
        if self.control_points.is_empty() {
            0
        } else {
            self.control_points[0].len()
        }
    }

    /// Evaluate the surface at parameter (u, v), returning a 3D point.
    pub fn evaluate(&self, u: f32, v: f32) -> Vec3 {
        let u_span = find_span(self.u_degree, &self.u_knots, u);
        let v_span = find_span(self.v_degree, &self.v_knots, v);

        let mut sw = Vec3::ZERO;
        let mut w_sum = 0.0;

        for i in 0..=self.u_degree {
            let ui = u_span.saturating_sub(self.u_degree) + i;
            if ui >= self.u_count() {
                continue;
            }
            let nu = bspline_basis(ui, self.u_degree, u, &self.u_knots);

            for j in 0..=self.v_degree {
                let vj = v_span.saturating_sub(self.v_degree) + j;
                if vj >= self.v_count() {
                    continue;
                }
                let nv = bspline_basis(vj, self.v_degree, v, &self.v_knots);

                let cp = &self.control_points[ui][vj];
                let w = cp[3] * nu * nv;
                sw += Vec3::new(cp[0], cp[1], cp[2]) * w;
                w_sum += w;
            }
        }

        if w_sum.abs() > 1e-10 {
            sw / w_sum
        } else {
            sw
        }
    }

    /// Evaluate the surface normal at (u, v) via cross product of partial derivatives.
    pub fn normal(&self, u: f32, v: f32) -> Vec3 {
        let eps = 1e-4;
        let p0 = self.evaluate((u - eps).max(0.0), v);
        let p1 = self.evaluate((u + eps).min(1.0), v);
        let du = p1 - p0;

        let q0 = self.evaluate(u, (v - eps).max(0.0));
        let q1 = self.evaluate(u, (v + eps).min(1.0));
        let dv = q1 - q0;

        let n = du.cross(dv);
        if n.length() > 1e-10 {
            n.normalize()
        } else {
            Vec3::Y
        }
    }

    /// Uniform tessellation into a TriangleMesh.
    pub fn tessellate_uniform(
        &self,
        u_samples: usize,
        v_samples: usize,
    ) -> rc3d_mesh::TriangleMesh {
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        for i in 0..=u_samples {
            for j in 0..=v_samples {
                let u = i as f32 / u_samples as f32;
                let v = j as f32 / v_samples as f32;
                positions.push(self.evaluate(u, v));
            }
        }

        let stride = v_samples + 1;
        for i in 0..u_samples {
            for j in 0..v_samples {
                let a = (i * stride + j) as u32;
                let b = a + stride as u32;
                let c = a + 1;
                let d = b + 1;
                indices.extend_from_slice(&[a, b, c]);
                indices.extend_from_slice(&[c, b, d]);
            }
        }

        rc3d_mesh::TriangleMesh::from_indexed(&positions, &indices)
    }

    /// Uniform tessellation with analytical per-vertex normals computed from the NURBS surface.
    /// Returns (positions, normals, triangle_indices) where normals[i] = surface.normal(u_i, v_j).
    pub fn tessellate_uniform_with_normals(
        &self,
        u_samples: usize,
        v_samples: usize,
    ) -> TessellatedSurface {
        let stride = v_samples + 1;
        let mut positions = Vec::with_capacity((u_samples + 1) * (v_samples + 1));
        let mut normals = Vec::with_capacity(positions.capacity());
        let mut indices = Vec::with_capacity(u_samples * v_samples * 6);

        for i in 0..=u_samples {
            let u = i as f32 / u_samples as f32;
            for j in 0..=v_samples {
                let v = j as f32 / v_samples as f32;
                positions.push(self.evaluate(u, v));
                normals.push(self.normal(u, v));
            }
        }

        for i in 0..u_samples {
            for j in 0..v_samples {
                let a = (i * stride + j) as u32;
                let b = a + stride as u32;
                let c = a + 1;
                let d = b + 1;
                indices.extend_from_slice(&[a, b, c]);
                indices.extend_from_slice(&[c, b, d]);
            }
        }

        TessellatedSurface { positions, normals, indices }
    }

    /// Adaptive tessellation driven by curvature.
    pub fn tessellate_adaptive(&self, tolerance: f32) -> rc3d_mesh::TriangleMesh {
        tessellate_surface_adaptive(self, tolerance)
    }

    /// Screen-space adaptive tessellation with analytical normals.
    ///
    /// Subdivides quads until every projected edge is shorter than `max_px` pixels
    /// and the normal deviation between corners is below `angle_tol` radians.
    /// Produces view-dependent triangle density: fine up close, coarse far away.
    ///
    /// `mvp` is the model-view-projection matrix, `viewport` is (width, height) in pixels.
    pub fn tessellate_screen_space(
        &self,
        mvp: &glam::Mat4,
        viewport: (f32, f32),
        camera_pos: glam::Vec3,
        light_dir: glam::Vec3,
        max_px: f32,
        silhouette_px: f32,
        terminator_px: f32,
        angle_tol: f32,
        max_depth: usize,
    ) -> TessellatedSurface {
        const VERTEX_BUDGET: usize = 50_000;
        let initial = 6usize;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();

        // Seed vertices at coarse grid
        let n = initial + 1;
        for i in 0..n {
            let u = i as f32 / initial as f32;
            for j in 0..n {
                let v = j as f32 / initial as f32;
                positions.push(self.evaluate(u, v));
                normals.push(self.normal(u, v));
            }
        }

        // Recursively subdivide each quad (budget-limited)
        let stride = n;
        for ci in 0..initial {
            for cj in 0..initial {
                let u0 = ci as f32 / initial as f32;
                let u1 = (ci + 1) as f32 / initial as f32;
                let v0 = cj as f32 / initial as f32;
                let v1 = (cj + 1) as f32 / initial as f32;
                let i00 = ci * stride + cj;
                let i10 = (ci + 1) * stride + cj;
                let i01 = ci * stride + cj + 1;
                let i11 = (ci + 1) * stride + cj + 1;

                subdivide_quad_screen(
                    self,
                    &mut positions,
                    &mut normals,
                    &mut indices,
                    u0, u1, v0, v1,
                    i00, i10, i01, i11,
                    mvp, viewport, camera_pos, light_dir,
                    max_px, silhouette_px, terminator_px, angle_tol,
                    0, max_depth, VERTEX_BUDGET,
                );
            }
        }

        TessellatedSurface { positions, normals, indices }
    }
}

/// Output of [`NurbsSurface::tessellate_uniform_with_normals`].
pub struct TessellatedSurface {
    pub positions: Vec<glam::Vec3>,
    pub normals: Vec<glam::Vec3>,
    pub indices: Vec<u32>,
}

/// Recursively subdivide a quad region based on screen-space edge length and normal deviation.
///
/// Three special regions get tighter tolerances:
/// - **Silhouette** (N·view changes sign): `silhouette_px`
/// - **Terminator** (N·light → 0 or sign change): `terminator_px`
/// - **Interior** (faces camera + lit): `max_px`
#[allow(clippy::too_many_arguments)]
fn subdivide_quad_screen(
    surface: &NurbsSurface,
    positions: &mut Vec<glam::Vec3>,
    normals: &mut Vec<glam::Vec3>,
    indices: &mut Vec<u32>,
    u0: f32, u1: f32, v0: f32, v1: f32,
    idx00: usize, idx10: usize, idx01: usize, idx11: usize,
    mvp: &glam::Mat4,
    viewport: (f32, f32),
    camera_pos: glam::Vec3,
    light_dir: glam::Vec3,
    max_px: f32,
    silhouette_px: f32,
    terminator_px: f32,
    angle_tol: f32,
    depth: usize,
    max_depth: usize,
    vertex_budget: usize,
) {
    if depth >= max_depth || positions.len() >= vertex_budget {
        emit_quad(indices, idx00, idx10, idx01, idx11);
        return;
    }

    let p00 = positions[idx00]; let p10 = positions[idx10];
    let p01 = positions[idx01]; let p11 = positions[idx11];
    let n00 = normals[idx00]; let n10 = normals[idx10];
    let n01 = normals[idx01]; let n11 = normals[idx11];

    // ════ Phase 1: Screen-space size (cheapest check — no evaluate calls) ════
    let proj = |p: glam::Vec3| -> (f32, f32) {
        let clip = *mvp * p.extend(1.0);
        if clip.w.abs() < 1e-12 { return (0.0, 0.0); }
        let ndc = clip / clip.w;
        ((ndc.x * 0.5 + 0.5) * viewport.0, ((1.0 - ndc.y) * 0.5) * viewport.1)
    };
    let (s00x, s00y) = proj(p00);
    let (s10x, s10y) = proj(p10);
    let (s01x, s01y) = proj(p01);
    let (s11x, s11y) = proj(p11);

    let edge_len = |x1: f32, y1: f32, x2: f32, y2: f32| -> f32 {
        ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)).sqrt()
    };
    let max_edge = edge_len(s00x, s00y, s10x, s10y)
        .max(edge_len(s10x, s10y, s11x, s11y))
        .max(edge_len(s11x, s11y, s01x, s01y))
        .max(edge_len(s01x, s01y, s00x, s00y));

    // Sub-pixel quads: never subdivide — invisible improvement.
    const MIN_SUBDIV_PX: f32 = 2.0;
    if max_edge < MIN_SUBDIV_PX {
        emit_quad(indices, idx00, idx10, idx01, idx11);
        return;
    }

    // ════ Phase 2: Corner normal deviation (cheap — data already loaded) ════
    let avg_corner = (n00 + n10 + n01 + n11).normalize();
    let corner_angle = n00.dot(avg_corner).clamp(-1.0, 1.0).acos()
        .max(n10.dot(avg_corner).clamp(-1.0, 1.0).acos())
        .max(n01.dot(avg_corner).clamp(-1.0, 1.0).acos())
        .max(n11.dot(avg_corner).clamp(-1.0, 1.0).acos());

    // ════ Phase 3: Classify region for adaptive tolerance ════
    let dot00 = n00.dot((camera_pos - p00).normalize());
    let dot10 = n10.dot((camera_pos - p10).normalize());
    let dot01 = n01.dot((camera_pos - p01).normalize());
    let dot11 = n11.dot((camera_pos - p11).normalize());
    let is_silhouette = (dot00 > 0.0 || dot10 > 0.0 || dot01 > 0.0 || dot11 > 0.0)
        && (dot00 < 0.0 || dot10 < 0.0 || dot01 < 0.0 || dot11 < 0.0);

    let l00 = n00.dot(light_dir); let l10 = n10.dot(light_dir);
    let l01 = n01.dot(light_dir); let l11 = n11.dot(light_dir);
    let is_terminator = (l00 > 0.0 || l10 > 0.0 || l01 > 0.0 || l11 > 0.0)
        && (l00 < 0.0 || l10 < 0.0 || l01 < 0.0 || l11 < 0.0)
        || l00.abs() < 0.15 || l10.abs() < 0.15 || l01.abs() < 0.15 || l11.abs() < 0.15;

    let effective_px = if is_silhouette {
        silhouette_px
    } else if is_terminator {
        terminator_px
    } else {
        max_px
    };

    // Fast decision: if edge or corner deviation already triggers, skip expensive checks.
    let fast_subdiv = max_edge > effective_px || corner_angle > angle_tol;

    // ════ Phase 4: Expensive center check — ONLY for ambiguous quads ════
    // A quad is "ambiguous" when corner checks pass but the quad is large enough
    // that a fold could hide in the interior (edges > 4px and corner deviation low).
    let um = (u0 + u1) * 0.5;
    let vm = (v0 + v1) * 0.5;
    let needs_subdiv = if fast_subdiv {
        true
    } else if max_edge > MIN_SUBDIV_PX * 2.0 {
        // Only probe the center for quads that are at least moderately large on screen
        let p_center = surface.evaluate(um, vm);
        let bilinear_center = (p00 + p10 + p01 + p11) * 0.25;
        let displacement = (p_center - bilinear_center).length();
        let quad_diag = (p11 - p00).length().max(1e-10);
        if displacement / quad_diag > 0.01 {
            true
        } else {
            let n_center = surface.normal(um, vm);
            let avg_with_center = (n00 + n10 + n01 + n11 + n_center).normalize();
            n_center.dot(avg_with_center).clamp(-1.0, 1.0).acos() > angle_tol
        }
    } else {
        false
    };

    if !needs_subdiv {
        emit_quad(indices, idx00, idx10, idx01, idx11);
        return;
    }

    // ════ Phase 5: Subdivide — evaluate midpoints ════
    let p_center = surface.evaluate(um, vm);
    let n_center = surface.normal(um, vm);
    let p_um0 = surface.evaluate(um, v0);
    let p_um1 = surface.evaluate(um, v1);
    let p_0vm = surface.evaluate(u0, vm);
    let p_1vm = surface.evaluate(u1, vm);

    let n_um0 = surface.normal(um, v0);
    let n_um1 = surface.normal(um, v1);
    let n_0vm = surface.normal(u0, vm);
    let n_1vm = surface.normal(u1, vm);

    let idx_um0 = positions.len(); positions.push(p_um0); normals.push(n_um0);
    let idx_um1 = positions.len(); positions.push(p_um1); normals.push(n_um1);
    let idx_0vm = positions.len(); positions.push(p_0vm); normals.push(n_0vm);
    let idx_1vm = positions.len(); positions.push(p_1vm); normals.push(n_1vm);
    let idx_center = positions.len(); positions.push(p_center); normals.push(n_center);

    let d = depth + 1;
    subdivide_quad_screen(surface, positions, normals, indices, u0, um, v0, vm, idx00, idx_um0, idx_0vm, idx_center, mvp, viewport, camera_pos, light_dir, max_px, silhouette_px, terminator_px, angle_tol, d, max_depth, vertex_budget);
    subdivide_quad_screen(surface, positions, normals, indices, um, u1, v0, vm, idx_um0, idx10, idx_center, idx_1vm, mvp, viewport, camera_pos, light_dir, max_px, silhouette_px, terminator_px, angle_tol, d, max_depth, vertex_budget);
    subdivide_quad_screen(surface, positions, normals, indices, u0, um, vm, v1, idx_0vm, idx_center, idx01, idx_um1, mvp, viewport, camera_pos, light_dir, max_px, silhouette_px, terminator_px, angle_tol, d, max_depth, vertex_budget);
    subdivide_quad_screen(surface, positions, normals, indices, um, u1, vm, v1, idx_center, idx_1vm, idx_um1, idx11, mvp, viewport, camera_pos, light_dir, max_px, silhouette_px, terminator_px, angle_tol, d, max_depth, vertex_budget);
}

fn emit_quad(indices: &mut Vec<u32>, a: usize, b: usize, c: usize, d: usize) {
    let a = a as u32; let b = b as u32;
    let c = c as u32; let d = d as u32;
    indices.extend_from_slice(&[a, b, d]);
    indices.extend_from_slice(&[a, d, c]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_square() {
        let grid: Vec<Vec<Vec3>> = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| Vec3::new(i as f32, j as f32, 0.0))
                    .collect()
            })
            .collect();
        let surface = NurbsSurface::from_points_grid(&grid, 2, 2);
        let p = surface.evaluate(0.5, 0.5);
        assert!((p.x - 1.0).abs() < 0.01, "x={}", p.x);
        assert!((p.y - 1.0).abs() < 0.01, "y={}", p.y);
        assert!(p.z.abs() < 0.01, "z={}", p.z);
    }

    #[test]
    fn test_normal_unit_length() {
        let grid: Vec<Vec<Vec3>> = (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| Vec3::new(i as f32, j as f32, (i * j) as f32 * 0.3))
                    .collect()
            })
            .collect();
        let surface = NurbsSurface::from_points_grid(&grid, 3, 3);
        let n = surface.normal(0.5, 0.5);
        assert!((n.length() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_tessellate_uniform_triangle_count() {
        let grid: Vec<Vec<Vec3>> = (0..3)
            .map(|i| (0..3).map(|j| Vec3::new(i as f32, j as f32, 0.0)).collect())
            .collect();
        let surface = NurbsSurface::from_points_grid(&grid, 2, 2);
        let mesh = surface.tessellate_uniform(4, 4);
        // 4x4 quads = 16 quads × 2 triangles = 32 triangles, 25 vertices
        assert_eq!(mesh.positions.len(), 25);
        assert_eq!(mesh.tri_indices.len(), 96); // 32 * 3
    }
}
