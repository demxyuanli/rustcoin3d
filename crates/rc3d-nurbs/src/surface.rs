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

    /// Adaptive tessellation driven by curvature.
    pub fn tessellate_adaptive(&self, tolerance: f32) -> rc3d_mesh::TriangleMesh {
        tessellate_surface_adaptive(self, tolerance)
    }
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
