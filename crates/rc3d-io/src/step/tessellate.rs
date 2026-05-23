//! Face tessellation: convert step faces to triangle meshes.

use std::collections::HashMap;
use rc3d_core::math::Vec3;
use super::topology::{StepFace, StepEdge};
use super::parser::EntityIndex;
use super::pcurve::FaceTrim;
use super::surface_tess;
use super::geom;

#[derive(Debug, Default)]
pub struct MeshResult {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub indices: Vec<i32>,
}

impl MeshResult {
    /// Compute per-vertex normals by averaging normals of adjacent triangles.
    /// Call after vertices and indices are populated.
    pub fn compute_normals(&mut self) {
        if self.vertices.is_empty() || self.indices.is_empty() {
            return;
        }

        let mut normals_acc = vec![Vec3::ZERO; self.vertices.len()];

        // Each triangle is stored as 4 indices: i0, i1, i2, -1 (sentinel)
        for chunk in self.indices.chunks(4) {
            if chunk.len() < 3 {
                continue;
            }
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len() {
                continue;
            }

            let v0 = self.vertices[i0];
            let v1 = self.vertices[i1];
            let v2 = self.vertices[i2];

            let n = (v1 - v0).cross(v2 - v0);
            normals_acc[i0] = normals_acc[i0] + n;
            normals_acc[i1] = normals_acc[i1] + n;
            normals_acc[i2] = normals_acc[i2] + n;
        }

        self.normals = normals_acc
            .into_iter()
            .map(|n| {
                let len = n.length();
                if len > 1e-10 {
                    n * (1.0 / len)
                } else {
                    Vec3::Z // default normal for degenerate vertices
                }
            })
            .collect();
    }

    /// Finalize normals: prefer analytical surface normals where available,
    /// fall back to face-averaged normals for any missing entries.
    pub fn finalize_normals(&mut self) {
        if self.vertices.is_empty() || self.indices.is_empty() {
            return;
        }

        // Ensure normals array exists
        if self.normals.len() != self.vertices.len() {
            self.normals = vec![Vec3::ZERO; self.vertices.len()];
        }

        // Compute face-averaged normals into a temporary buffer
        let mut face_normals = vec![Vec3::ZERO; self.vertices.len()];
        for chunk in self.indices.chunks(4) {
            if chunk.len() < 3 {
                continue;
            }
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len() {
                continue;
            }
            let n = (self.vertices[i1] - self.vertices[i0]).cross(self.vertices[i2] - self.vertices[i0]);
            face_normals[i0] = face_normals[i0] + n;
            face_normals[i1] = face_normals[i1] + n;
            face_normals[i2] = face_normals[i2] + n;
        }

        // Prefer analytical normals; use face-average as fallback
        for i in 0..self.vertices.len() {
            let ana = self.normals[i];
            if ana.length() > 1e-10 {
                self.normals[i] = ana.normalize();
            } else {
                let n = face_normals[i];
                self.normals[i] = if n.length() > 1e-10 {
                    n.normalize()
                } else {
                    Vec3::Z
                };
            }
        }
    }
}

/// Tessellate all faces into a single mesh, with optional per-face trim data.
pub fn tessellate_faces(
    faces: &[StepFace],
    entities: &EntityIndex,
) -> MeshResult {
    tessellate_faces_with_trim(faces, entities, &[])
}

/// Tessellate faces with optional per-face PCURVE trim data.
pub fn tessellate_faces_with_trim(
    faces: &[StepFace],
    entities: &EntityIndex,
    trims: &[Option<FaceTrim>],
) -> MeshResult {
    let mut result = MeshResult::default();
    let mut pos_map: HashMap<[u32; 3], i32> = HashMap::new();

    for (i, face) in faces.iter().enumerate() {
        let trim = trims.get(i).and_then(|t| t.as_ref());
        tessellate_face(face, entities, trim, &mut result, &mut pos_map);
    }

    result.finalize_normals();
    result
}

fn tessellate_face(
    face: &StepFace,
    entities: &EntityIndex,
    _trim: Option<&FaceTrim>,  // trim is used inside surface_tess::tessellate_curved_face via pcurve::extract_face_trim
    mesh: &mut MeshResult,
    pos_map: &mut HashMap<[u32; 3], i32>,
) {
    // Try surface tessellation first
    if face.surface_id.is_some() {
        let trim = super::pcurve::extract_face_trim(face, entities);
        if let Some(surf_mesh) = surface_tess::tessellate_curved_face(face, entities, trim.as_ref()) {
            if !surf_mesh.vertices.is_empty() {
                // Build remap from surf_mesh local indices to global mesh indices
                let mut remap: Vec<i32> = vec![-1; surf_mesh.vertices.len()];
                for (local_idx, v) in surf_mesh.vertices.iter().enumerate() {
                    let hash = rc3d_core::utils::hash::f32x3_quantized_bits([v.x, v.y, v.z]);
                    let global_idx = if let Some(&idx) = pos_map.get(&hash) {
                        idx
                    } else {
                        let i = mesh.vertices.len() as i32;
                        mesh.vertices.push(*v);
                        // Reserve slot for normal (will be filled below if available)
                        mesh.normals.push(Vec3::ZERO);
                        pos_map.insert(hash, i);
                        i
                    };
                    remap[local_idx] = global_idx;
                }

                // Merge analytical normals from NURBS surface evaluation.
                // Multiple local vertices may map to the same global vertex
                // (via hash dedup), so accumulate and normalize later.
                while mesh.normals.len() < mesh.vertices.len() {
                    mesh.normals.push(Vec3::ZERO);
                }
                if !surf_mesh.normals.is_empty() {
                    for (local_idx, n) in surf_mesh.normals.iter().enumerate() {
                        let gi = remap[local_idx] as usize;
                        if gi < mesh.normals.len() {
                            mesh.normals[gi] = mesh.normals[gi] + *n;
                        }
                    }
                }

                // Merge indices (each triangle = 3 indices + -1 sentinel)
                for chunk in surf_mesh.indices.chunks(4) {
                    if chunk.len() == 4 && chunk[3] == -1 {
                        for &local_idx in &chunk[..3] {
                            mesh.indices.push(if local_idx >= 0 && (local_idx as usize) < remap.len() {
                                remap[local_idx as usize]
                            } else {
                                -1
                            });
                        }
                        mesh.indices.push(-1);
                    }
                }
                return;
            }
        }
    }

    // Fallback: edge-loop fan triangulation
    for bloop in &face.bounds {
        let mut loop_vertices: Vec<i32> = Vec::new();

        for edge in &bloop.edges {
            // Sample curve between start and end
            let pts = geom::sample_curve(edge.curve_id, entities, edge.start, edge.end, 0.1);

            // Add all points except the last (which equals the next edge's first point)
            let n = pts.len();
            for (j, pt) in pts.iter().enumerate() {
                if j == n - 1 && edge_is_last_in_loop(edge, bloop, j) {
                    // Last point of last edge = first point of first edge (closed loop)
                    continue;
                }
                let hash = rc3d_core::utils::hash::f32x3_quantized_bits([pt.x, pt.y, pt.z]);
                let idx = if let Some(&i) = pos_map.get(&hash) {
                    i
                } else {
                    let i = mesh.vertices.len() as i32;
                    mesh.vertices.push(*pt);
                    pos_map.insert(hash, i);
                    i
                };
                loop_vertices.push(idx);
            }
        }

        if loop_vertices.len() >= 3 {
            triangulate_polygon(&loop_vertices, &mesh.vertices, &mut mesh.indices);
        }
    }
}

fn edge_is_last_in_loop(_edge: &StepEdge, _loop: &super::topology::StepLoop, _index: usize) -> bool {
    false // We let duplicate removal handle this via the hash map
}

/// Triangulate a polygon defined by vertex indices using earcutr (ear clipping).
/// Supports both convex and concave polygons correctly.
fn triangulate_polygon(poly_indices: &[i32], vertices: &[Vec3], out: &mut Vec<i32>) {
    if poly_indices.len() < 3 {
        return;
    }

    // Collect 3D points from vertex indices
    let points: Vec<Vec3> = poly_indices.iter()
        .filter_map(|&idx| {
            let i = idx as usize;
            if i < vertices.len() { Some(vertices[i]) } else { None }
        })
        .collect();

    if points.len() < 3 {
        return;
    }

    // Compute polygon normal from first 3 non-collinear points
    let v0 = points[0];
    let v1 = points[1];
    // Find a third point that is not collinear with v0-v1
    let mut normal = Vec3::ZERO;
    let mut _v2_idx = 2;
    for (i, &p) in points.iter().enumerate().skip(2) {
        let n = (v1 - v0).cross(p - v0);
        if n.length() > 1e-10 {
            normal = n;
            _v2_idx = i;
            break;
        }
    }

    if normal.length() < 1e-10 {
        // Degenerate (collinear) polygon — fall back to fan
        triangulate_fan_fallback(poly_indices, out);
        return;
    }
    let normal = normal.normalize();

    // Choose projection axis (drop the axis with largest normal component)
    let abs_n = [normal.x.abs(), normal.y.abs(), normal.z.abs()];
    let drop_axis = if abs_n[0] >= abs_n[1] && abs_n[0] >= abs_n[2] { 0 }
    else if abs_n[1] >= abs_n[2] { 1 }
    else { 2 };

    // Project to 2D
    let flat: Vec<f64> = points.iter().flat_map(|p| {
        let (x, y) = match drop_axis {
            0 => (p.y as f64, p.z as f64),
            1 => (p.x as f64, p.z as f64),
            _ => (p.x as f64, p.y as f64),
        };
        [x, y]
    }).collect();

    // Run earcutr ear clipping
    let ear_indices = match earcutr::earcut(&flat, &[], 2) {
        Ok(indices) => indices,
        Err(_) => {
            // earcutr failed (e.g. degenerate polygon) — fall back to fan
            triangulate_fan_fallback(poly_indices, out);
            return;
        }
    };

    // Map earcutr indices back to mesh vertex indices
    for chunk in ear_indices.chunks(3) {
        if chunk.len() == 3 {
            let i0 = chunk[0];
            let i1 = chunk[1];
            let i2 = chunk[2];
            if i0 < poly_indices.len() && i1 < poly_indices.len() && i2 < poly_indices.len() {
                out.push(poly_indices[i0]);
                out.push(poly_indices[i1]);
                out.push(poly_indices[i2]);
                out.push(-1);
            }
        }
    }
}

/// Fallback fan triangulation for degenerate (collinear) polygons.
fn triangulate_fan_fallback(poly: &[i32], out: &mut Vec<i32>) {
    let v0 = poly[0];
    for j in 1..poly.len() - 1 {
        out.push(v0);
        out.push(poly[j]);
        out.push(poly[j + 1]);
        out.push(-1);
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser;
    use super::super::topology;
    use super::super::pcurve::FaceTrim;

    fn make_entities(data: &str) -> EntityIndex {
        let input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
            data
        );
        parser::parse_exchange(&input).unwrap().entities
    }

    #[test]
    fn test_tessellate_without_trim() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = FACE_SURFACE('', (#15));
#17 = CLOSED_SHELL('', (#16));
#20 = LINE('', #1, #2);\
",
        );
        let faces = topology::collect_shell_faces(&entities);
        assert!(!faces.is_empty(), "should resolve at least one face");
        let mesh = tessellate_faces(&faces, &entities);
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
    }

    #[test]
    fn test_tessellate_with_trim_noop() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#10 = EDGE_CURVE('', #1, #2, #20, .T.);
#11 = EDGE_CURVE('', #2, #3, #20, .T.);
#12 = EDGE_CURVE('', #3, #4, #20, .T.);
#13 = EDGE_CURVE('', #4, #1, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = FACE_SURFACE('', (#15));
#17 = CLOSED_SHELL('', (#16));
#20 = LINE('', #1, #2);\
",
        );
        let faces = topology::collect_shell_faces(&entities);
        assert!(!faces.is_empty(), "should find faces from shell");
        let trims: Vec<Option<FaceTrim>> = faces.iter().map(|_| None).collect();
        let mesh = tessellate_faces_with_trim(&faces, &entities, &trims);
        // Even without trim, tessellation works for planar face
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
    }

    #[test]
    fn test_earcutr_concave_polygon() {
        // L-shaped concave polygon — fan triangulation would produce
        // a triangle crossing the interior, earcutr should not.
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),  // 0
            Vec3::new(4.0, 0.0, 0.0),  // 1
            Vec3::new(4.0, 2.0, 0.0),  // 2
            Vec3::new(2.0, 2.0, 0.0),  // 3
            Vec3::new(2.0, 4.0, 0.0),  // 4
            Vec3::new(0.0, 4.0, 0.0),  // 5
        ];
        let poly_indices: Vec<i32> = vec![0, 1, 2, 3, 4, 5];
        let mut indices = Vec::new();
        triangulate_polygon(&poly_indices, &vertices, &mut indices);

        // Should produce at least 4 triangles (6-gon → 4 triangles)
        let tri_count = indices.chunks(4).filter(|c| c.len() == 4 && c[3] == -1).count();
        assert!(tri_count >= 4, "L-shape should have >= 4 triangles, got {}", tri_count);

        // Verify no triangle centroid falls outside the L-shape
        for chunk in indices.chunks(4) {
            if chunk.len() < 3 || chunk[3] != -1 { continue; }
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() { continue; }
            let centroid = (vertices[i0] + vertices[i1] + vertices[i2]) * (1.0 / 3.0);
            // Centroid must be inside the L-shape (0..4 x 0..4 minus the cut-out at 2..4 x 2..4)
            let in_l = centroid.x >= -0.01 && centroid.x <= 4.01
                && centroid.y >= -0.01 && centroid.y <= 4.01
                && !(centroid.x > 2.01 && centroid.y > 2.01);
            assert!(in_l, "Triangle centroid {:?} is outside the L-shape", centroid);
        }
    }
}
