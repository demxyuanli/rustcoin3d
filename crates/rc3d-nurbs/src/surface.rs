use glam::Vec3;
use crate::basis::bspline_basis;
use crate::knot::{find_span, open_uniform_knots};
use crate::tessellate::tessellate_surface_adaptive;

/// A NURBS surface defined by a grid of homogeneous control points.
#[derive(Clone, Debug)]
pub struct NurbsRenderSurface {
    pub control_points: Vec<Vec<[f32; 4]>>, // [u_count][v_count] = (wx, wy, wz, w)
    pub u_knots: Vec<f32>,
    pub v_knots: Vec<f32>,
    pub u_degree: usize,
    pub v_degree: usize,
}

/// Quality parameters for screen-space adaptive tessellation.
#[derive(Clone, Copy, Debug)]
pub struct TessQuality {
    pub max_px: f32,
    pub silhouette_px: f32,
    pub terminator_px: f32,
    pub angle_tol: f32,
    pub max_depth: usize,
}

impl NurbsRenderSurface {
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

    /// Evaluate the surface normal at (u, v) via analytical cross product.
    pub fn normal(&self, u: f32, v: f32) -> Vec3 {
        let (du, dv) = self.derivative(u, v);
        let n = du.cross(dv);
        if n.length() > 1e-10 {
            n.normalize()
        } else {
            // Degenerate — probe a small neighborhood
            let eps = 1e-3;
            let probes = [(u + eps, v), (u - eps, v), (u, v + eps), (u, v - eps)];
            let mut best = Vec3::Y;
            let mut best_len = 0.0;
            for (up, vp) in probes {
                if !(0.0..=1.0).contains(&up) || !(0.0..=1.0).contains(&vp) { continue; }
                let (du2, dv2) = self.derivative(up, vp);
                let n2 = du2.cross(dv2);
                let l2 = n2.length();
                if l2 > best_len { best_len = l2; best = n2 / l2.max(1e-12); }
            }
            best
        }
    }

    /// Analytical first-order partial derivatives ∂S/∂u and ∂S/∂v.
    ///
    /// Uses the analytical B-spline derivative recurrence (same formula as
    /// rc3d-shape::NurbsSurface::evaluate_with_derivative) applied to the
    /// homogeneous control points then dehomogenized via quotient rule.
    pub fn derivative(&self, u: f32, v: f32) -> (Vec3, Vec3) {
        use crate::basis::bspline_bases;
        use crate::knot::find_span;

        let u_span = find_span(self.u_degree, &self.u_knots, u);
        let v_span = find_span(self.v_degree, &self.v_knots, v);

        let basis_u = bspline_bases(u_span, self.u_degree, u, &self.u_knots);
        let basis_v = bspline_bases(v_span, self.v_degree, v, &self.v_knots);

        // Analytical first derivatives of basis functions
        let du_basis = analytical_basis_derivs(u_span, self.u_degree, u, &self.u_knots);
        let dv_basis = analytical_basis_derivs(v_span, self.v_degree, v, &self.v_knots);

        // Accumulate weighted sums for position and derivatives
        let mut sw = Vec3::ZERO;   let mut w_sum = 0.0;
        let mut sw_u = Vec3::ZERO; let mut w_u = 0.0;
        let mut sw_v = Vec3::ZERO; let mut w_v = 0.0;

        for &(i, nu) in &basis_u {
            let dn_du = lookup(&du_basis, i);
            for &(j, nv) in &basis_v {
                let cp = &self.control_points[i][j];
                let wgt = cp[3];
                let pw = Vec3::new(cp[0], cp[1], cp[2]);
                let dn_dv = lookup(&dv_basis, j);

                // Position weight
                let c = nu * nv * wgt;
                sw += pw * c;
                w_sum += c;

                // ∂/∂u
                let c_u = dn_du * nv * wgt;
                sw_u += pw * c_u;
                w_u += c_u;

                // ∂/∂v
                let c_v = nu * dn_dv * wgt;
                sw_v += pw * c_v;
                w_v += c_v;
            }
        }

        if w_sum.abs() < 1e-10 {
            return (Vec3::X, Vec3::Y);
        }

        let inv_w = 1.0 / w_sum;
        let inv_w2 = inv_w * inv_w;
        let du = (sw_u * w_sum - sw * w_u) * inv_w2;
        let dv = (sw_v * w_sum - sw * w_v) * inv_w2;

        // NaN guard
        if du.is_nan() || dv.is_nan() {
            return (Vec3::X, Vec3::Y);
        }
        (du, dv)
    }

    /// Extract an isoparametric edge as a NURBS curve for boundary matching.
    ///
    /// Returns the control points along the specified edge. For UMin/UMax the
    /// curve runs in the v direction; for VMin/VMax it runs in the u direction.
    pub fn boundary_curve(&self, edge: BoundaryEdge) -> crate::NurbsCurve {
        match edge {
            BoundaryEdge::UMin => {
                let cp: Vec<Vec3> = (0..self.v_count())
                    .map(|j| {
                        let p = &self.control_points[0][j];
                        Vec3::new(p[0] / p[3], p[1] / p[3], p[2] / p[3])
                    })
                    .collect();
                crate::NurbsCurve::from_points(&cp, self.v_degree)
            }
            BoundaryEdge::UMax => {
                let u_last = self.u_count() - 1;
                let cp: Vec<Vec3> = (0..self.v_count())
                    .map(|j| {
                        let p = &self.control_points[u_last][j];
                        Vec3::new(p[0] / p[3], p[1] / p[3], p[2] / p[3])
                    })
                    .collect();
                crate::NurbsCurve::from_points(&cp, self.v_degree)
            }
            BoundaryEdge::VMin => {
                let cp: Vec<Vec3> = (0..self.u_count())
                    .map(|i| {
                        let p = &self.control_points[i][0];
                        Vec3::new(p[0] / p[3], p[1] / p[3], p[2] / p[3])
                    })
                    .collect();
                crate::NurbsCurve::from_points(&cp, self.u_degree)
            }
            BoundaryEdge::VMax => {
                let v_last = self.v_count() - 1;
                let cp: Vec<Vec3> = (0..self.u_count())
                    .map(|i| {
                        let p = &self.control_points[i][v_last];
                        Vec3::new(p[0] / p[3], p[1] / p[3], p[2] / p[3])
                    })
                    .collect();
                crate::NurbsCurve::from_points(&cp, self.u_degree)
            }
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
    /// Subdivides quads until every projected edge is shorter than `quality.max_px` pixels
    /// and the normal deviation between corners is below `quality.angle_tol` radians.
    /// Produces view-dependent triangle density: fine up close, coarse far away.
    ///
    /// `mvp` is the model-view-projection matrix, `viewport` is (width, height) in pixels.
    pub fn tessellate_screen_space(
        &self,
        mvp: &glam::Mat4,
        viewport: (f32, f32),
        camera_pos: glam::Vec3,
        light_dir: glam::Vec3,
        quality: &TessQuality,
    ) -> TessellatedSurface {
        const VERTEX_BUDGET: usize = 80_000;
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

        // ── Progressive budget: weight each quad by its screen-space diagonal ──
        let proj_len = |p: glam::Vec3| -> f32 {
            let clip = *mvp * p.extend(1.0);
            if clip.w <= 1e-6 { return f32::MAX; } // near camera → large weight
            let ndc = clip / clip.w;
            ((ndc.x * viewport.0).powi(2) + ((1.0 - ndc.y) * viewport.1).powi(2)).sqrt()
        };
        let stride = n;
        let total_quads = initial * initial;
        let mut quad_weights: Vec<f32> = Vec::with_capacity(total_quads);
        let mut total_weight: f32 = 0.0;
        for ci in 0..initial {
            for cj in 0..initial {
                let i00 = ci * stride + cj;
                let i11 = (ci + 1) * stride + cj + 1;
                let ss_diag = proj_len(positions[i00]).max(proj_len(positions[i11]));
                let w = ss_diag.clamp(1.0, 1e6);
                quad_weights.push(w);
                total_weight += w;
            }
        }
        // Allocate per-quad budget proportionally; each quad gets at least 500 vertices
        let min_per_quad: usize = 500;
        let pool = VERTEX_BUDGET.saturating_sub(total_quads * min_per_quad);
        let scale = if total_weight > 0.0 { pool as f32 / total_weight } else { 0.0 };

        // Recursively subdivide each quad with its own budget
        let mut idx = 0;
        for ci in 0..initial {
            for cj in 0..initial {
                let quad_budget = min_per_quad + (quad_weights[idx] * scale) as usize;
                idx += 1;
                let u0 = ci as f32 / initial as f32;
                let u1 = (ci + 1) as f32 / initial as f32;
                let v0 = cj as f32 / initial as f32;
                let v1 = (cj + 1) as f32 / initial as f32;
                let i00 = ci * stride + cj;
                let i10 = (ci + 1) * stride + cj;
                let i01 = ci * stride + cj + 1;
                let i11 = (ci + 1) * stride + cj + 1;
                let mut used = 0usize;

                subdivide_quad_screen(
                    self,
                    &mut positions,
                    &mut normals,
                    &mut indices,
                    u0, u1, v0, v1,
                    i00, i10, i01, i11,
                    mvp, viewport, camera_pos, light_dir,
                    quality,
                    0, quad_budget,
                    &mut used,
                );
            }
        }

        TessellatedSurface { positions, normals, indices }
    }
}

/// Output of [`NurbsRenderSurface::tessellate_uniform_with_normals`].
/// Which isoparametric edge of a NURBS surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryEdge {
    UMin, // u = 0
    UMax, // u = 1
    VMin, // v = 0
    VMax, // v = 1
}

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
    surface: &NurbsRenderSurface,
    positions: &mut Vec<glam::Vec3>,
    normals: &mut Vec<glam::Vec3>,
    indices: &mut Vec<u32>,
    u0: f32, u1: f32, v0: f32, v1: f32,
    idx00: usize, idx10: usize, idx01: usize, idx11: usize,
    mvp: &glam::Mat4,
    viewport: (f32, f32),
    camera_pos: glam::Vec3,
    light_dir: glam::Vec3,
    quality: &TessQuality,
    depth: usize,
    vertex_budget: usize,
    vertices_used: &mut usize,
) {
    let p00 = positions[idx00]; let p10 = positions[idx10];
    let p01 = positions[idx01]; let p11 = positions[idx11];
    let n00 = normals[idx00]; let n10 = normals[idx10];
    let n01 = normals[idx01]; let n11 = normals[idx11];

    // ════ Phase 1: Screen-space size + camera-boundary detection ════
    let proj = |p: glam::Vec4| -> (f32, f32) {
        if p.w <= 1e-6 {
            return (f32::NEG_INFINITY, f32::NEG_INFINITY);
        }
        let ndc = p / p.w;
        ((ndc.x * 0.5 + 0.5) * viewport.0, ((1.0 - ndc.y) * 0.5) * viewport.1)
    };
    let clip00 = *mvp * p00.extend(1.0);
    let clip10 = *mvp * p10.extend(1.0);
    let clip01 = *mvp * p01.extend(1.0);
    let clip11 = *mvp * p11.extend(1.0);

    // Quad straddles the camera plane: some corners behind, some in front.
    // Force subdivision — screen-space edge heuristic is unreliable here.
    let crosses_camera = clip00.w <= 1e-6 || clip10.w <= 1e-6
        || clip01.w <= 1e-6 || clip11.w <= 1e-6;

    // Camera-crossing quads get double the per-quad budget since they need
    // finer tessellation near the camera boundary to avoid visible clipping.
    let effective_budget = if crosses_camera { vertex_budget * 2 } else { vertex_budget };
    if depth >= quality.max_depth || *vertices_used >= effective_budget {
        emit_quad(indices, idx00, idx10, idx01, idx11);
        return;
    }

    let (s00x, s00y) = proj(clip00);
    let (s10x, s10y) = proj(clip10);
    let (s01x, s01y) = proj(clip01);
    let (s11x, s11y) = proj(clip11);

    // ── Viewport culling: skip quads fully outside the viewport ──
    // Recover budget for visible quads; off-screen quads contribute nothing.
    let sx_min = s00x.min(s10x).min(s01x).min(s11x);
    let sx_max = s00x.max(s10x).max(s01x).max(s11x);
    let sy_min = s00y.min(s10y).min(s01y).min(s11y);
    let sy_max = s00y.max(s10y).max(s01y).max(s11y);
    let vw = viewport.0;
    let vh = viewport.1;
    if sx_max < 0.0 || sx_min > vw || sy_max < 0.0 || sy_min > vh {
        emit_quad(indices, idx00, idx10, idx01, idx11);
        return;
    }

    let edge_len = |x1: f32, y1: f32, x2: f32, y2: f32| -> f32 {
        if x1.is_infinite() || x2.is_infinite() { return f32::MAX; }
        ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)).sqrt()
    };
    let max_edge = edge_len(s00x, s00y, s10x, s10y)
        .max(edge_len(s10x, s10y, s11x, s11y))
        .max(edge_len(s11x, s11y, s01x, s01y))
        .max(edge_len(s01x, s01y, s00x, s00y));

    // Sub-pixel quads: never subdivide — invisible improvement.
    // Skip for camera-straddling quads (their screen-space sizes are unreliable).
    const MIN_SUBDIV_PX: f32 = 2.0;
    if !crosses_camera && max_edge < MIN_SUBDIV_PX {
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
        quality.silhouette_px
    } else if is_terminator {
        quality.terminator_px
    } else {
        quality.max_px
    };

    // Fast decision: if edge or corner deviation already triggers, skip expensive checks.
    let fast_subdiv = max_edge > effective_px || corner_angle > quality.angle_tol;

    // ════ Phase 4: Expensive center check — ONLY for ambiguous quads ════
    // A quad is "ambiguous" when corner checks pass but the quad is large enough
    // that a fold could hide in the interior (edges > 4px and corner deviation low).
    let um = (u0 + u1) * 0.5;
    let vm = (v0 + v1) * 0.5;
    // Cache center evaluation for reuse in Phase 5 if subdivision is needed.
    let mut cached_p_center = None;
    let mut cached_n_center = None;
    let needs_subdiv = if fast_subdiv || crosses_camera {
        true
    } else if max_edge > MIN_SUBDIV_PX * 2.0 {
        let p_center = surface.evaluate(um, vm);
        cached_p_center = Some(p_center);
        let bilinear_center = (p00 + p10 + p01 + p11) * 0.25;
        let displacement = (p_center - bilinear_center).length();
        let quad_diag = (p11 - p00).length().max(1e-10);
        if displacement / quad_diag > 0.01 {
            true
        } else {
            let n_center = surface.normal(um, vm);
            cached_n_center = Some(n_center);
            let avg_with_center = (n00 + n10 + n01 + n11 + n_center).normalize();
            n_center.dot(avg_with_center).clamp(-1.0, 1.0).acos() > quality.angle_tol
        }
    } else {
        false
    };

    if !needs_subdiv {
        emit_quad(indices, idx00, idx10, idx01, idx11);
        return;
    }

    // ════ Phase 5: Subdivide — evaluate midpoints (reuse cached center) ════
    let p_center = cached_p_center.unwrap_or_else(|| surface.evaluate(um, vm));
    let n_center = cached_n_center.unwrap_or_else(|| surface.normal(um, vm));
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
    *vertices_used += 5;

    let d = depth + 1;
    subdivide_quad_screen(surface, positions, normals, indices, u0, um, v0, vm, idx00, idx_um0, idx_0vm, idx_center, mvp, viewport, camera_pos, light_dir, quality, d, vertex_budget, vertices_used);
    subdivide_quad_screen(surface, positions, normals, indices, um, u1, v0, vm, idx_um0, idx10, idx_center, idx_1vm, mvp, viewport, camera_pos, light_dir, quality, d, vertex_budget, vertices_used);
    subdivide_quad_screen(surface, positions, normals, indices, u0, um, vm, v1, idx_0vm, idx_center, idx01, idx_um1, mvp, viewport, camera_pos, light_dir, quality, d, vertex_budget, vertices_used);
    subdivide_quad_screen(surface, positions, normals, indices, um, u1, vm, v1, idx_center, idx_1vm, idx_um1, idx11, mvp, viewport, camera_pos, light_dir, quality, d, vertex_budget, vertices_used);
}

fn emit_quad(indices: &mut Vec<u32>, a: usize, b: usize, c: usize, d: usize) {
    let a = a as u32; let b = b as u32;
    let c = c as u32; let d = d as u32;
    indices.extend_from_slice(&[a, b, d]);
    indices.extend_from_slice(&[a, d, c]);
}

// ── Analytical basis derivative helpers ──────────────────────────

/// Linear scan lookup in a short basis derivative list.
#[inline]
fn lookup(basis: &[(usize, f32)], idx: usize) -> f32 {
    for &(i, v) in basis {
        if i == idx { return v; }
    }
    0.0
}

/// Compute first-order B-spline basis derivatives.
///
/// N'_{i,p}(t) = p/(k_{i+p}-k_i) * N_{i,p-1}(t) - p/(k_{i+p+1}-k_{i+1}) * N_{i+1,p-1}(t)
///
/// (Piegl & Tiller, The NURBS Book, Eq. 2.10)
fn analytical_basis_derivs(
    span: usize, degree: usize, t: f32, knots: &[f32],
) -> Vec<(usize, f32)> {
    use crate::basis::bspline_bases;

    if degree == 0 {
        return vec![(span, 0.0)];
    }

    let lower_bases = bspline_bases(span, degree - 1, t, knots);
    let p = degree as f32;
    let mut result = Vec::with_capacity(degree + 1);

    for i in span.saturating_sub(degree)..=span {
        let n_i = if i >= span.saturating_sub(degree - 1) {
            lookup(&lower_bases, i)
        } else { 0.0 };
        let n_ip1 = if i < span {
            lookup(&lower_bases, i + 1)
        } else { 0.0 };

        let d1 = knots[i + degree] - knots[i];
        let d2 = knots[i + degree + 1] - knots[i + 1];
        let t1 = if d1.abs() > 1e-12 { n_i / d1 } else { 0.0 };
        let t2 = if d2.abs() > 1e-12 { n_ip1 / d2 } else { 0.0 };

        let deriv = p * (t1 - t2);
        if deriv.abs() > 1e-12 {
            result.push((i, deriv));
        }
    }
    result
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
        let surface = NurbsRenderSurface::from_points_grid(&grid, 2, 2);
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
        let surface = NurbsRenderSurface::from_points_grid(&grid, 3, 3);
        let n = surface.normal(0.5, 0.5);
        assert!((n.length() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_tessellate_uniform_triangle_count() {
        let grid: Vec<Vec<Vec3>> = (0..3)
            .map(|i| (0..3).map(|j| Vec3::new(i as f32, j as f32, 0.0)).collect())
            .collect();
        let surface = NurbsRenderSurface::from_points_grid(&grid, 2, 2);
        let mesh = surface.tessellate_uniform(4, 4);
        // 4x4 quads = 16 quads × 2 triangles = 32 triangles, 25 vertices
        assert_eq!(mesh.positions.len(), 25);
        assert_eq!(mesh.tri_indices.len(), 96); // 32 * 3
    }
}
