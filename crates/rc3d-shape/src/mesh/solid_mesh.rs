//! Solid-level mesh assembly (OCC `BREP_WITH_VOIDS` semantics).
//!
//! Strategy (OCC BRepAlgoAPI_Cut equivalent):
//! 1. Mesh the outer shell normally.
//! 2. For each void face, attempt to project its UV boundary as an inner wire
//!    onto a matching outer face (face-level subtraction).
//! 3. If face-level projection isn't possible (surface mismatch), fall back to
//!    mesh-level void mesh append with reversed winding + vertex weld.

use rc3d_core::math::Real;
use std::collections::HashMap;

use crate::geom::SurfaceGeom;
use crate::mesh_result::MeshResult;
use crate::mesh_split::FaceTriRange;
use crate::store::BRepStore;
use crate::topo::{FaceKey, SolidKey};

use super::config::BRepMeshConfig;
use super::report::ShellMeshReport;
use super::shell_mesh::mesh_brep_shell_with_report;

#[derive(Debug)]
pub struct SolidMeshOutput {
    pub mesh: MeshResult,
    pub face_tri_ranges: HashMap<FaceKey, FaceTriRange>,
    pub report: ShellMeshReport,
}

fn face_ranges_from_report(
    report: &ShellMeshReport,
    tri_offset: usize,
) -> HashMap<FaceKey, FaceTriRange> {
    report
        .faces
        .iter()
        .filter(|f| f.tri_count > 0)
        .map(|f| {
            (
                f.face_key,
                FaceTriRange {
                    first_tri: f.first_tri + tri_offset,
                    tri_count: f.tri_count,
                },
            )
        })
        .collect()
}

fn merge_shell_report(acc: &mut ShellMeshReport, shell: &ShellMeshReport) {
    acc.face_count += shell.face_count;
    acc.meshed_faces += shell.meshed_faces;
    acc.grid_fallback_count += shell.grid_fallback_count;
    acc.cdt_constraint_failure_count += shell.cdt_constraint_failure_count;
    acc.total_tris += shell.total_tris;
    acc.shell_diag = acc.shell_diag.max(shell.shell_diag);
    acc.max_equiv_edge_weld_gap = acc
        .max_equiv_edge_weld_gap
        .max(shell.max_equiv_edge_weld_gap);
    acc.faces.extend_from_slice(&shell.faces);
}

/// Project a void face's wire edges onto an outer face's UV domain.
///
/// Returns inner UV wire vertices suitable for CDT hole constraints,
/// or `None` if projection fails (non-planar, no matching surface, etc.).
#[allow(dead_code)] // Phase 3 integration into mesh_solid_with_voids
fn project_void_face_as_inner_wire(
    store: &BRepStore,
    outer_face_key: FaceKey,
    void_face_key: FaceKey,
    tolerance: Real,
) -> Option<Vec<(Real, Real)>> {
    let outer_face = store.faces.get(outer_face_key)?;
    let void_face = store.faces.get(void_face_key)?;

    // Only attempt projection when both faces share the same surface type.
    // OCC Cut: faces must be co-planar or on the same surface.
    if std::mem::discriminant(&outer_face.surface) != std::mem::discriminant(&void_face.surface) {
        return None;
    }

    // Check co-planarity for planes (most common void case).
    if let (SurfaceGeom::Plane { origin: o1, normal: n1, .. },
            SurfaceGeom::Plane { origin: o2, normal: n2, .. }) = (&outer_face.surface, &void_face.surface)
    {
        if (n1.dot(*n2)).abs() < 0.999 || (o1 - o2).dot(*n1).abs() > tolerance * 10.0 {
            return None;
        }
    }

    let void_wire = store.wires.get(void_face.outer_wire)?;
    let mut projected_uvs: Vec<(Real, Real)> = Vec::new();

    for &(ek, _orient) in &void_wire.edges {
        let edge = store.edges.get(ek)?;
        // Get PCurve for the void face — this gives us the void wire UV on the void surface.
        let pc = edge.pcurves.get(&void_face_key)?;
        // Sample the PCurve midpoint in UV space.
        let uv_mid = pc.d0(0.5);
        // Map UV→3D on the void surface, then project 3D→UV on the outer surface.
        let pt_3d = void_face.surface.d0_native(uv_mid.0, uv_mid.1);
        let (u_outer, v_outer) = outer_face.surface.project(pt_3d)?;
        projected_uvs.push((u_outer, v_outer));
    }

    if projected_uvs.len() < 3 {
        return None;
    }

    // Close the loop.
    projected_uvs.push(projected_uvs[0]);
    Some(projected_uvs)
}

/// Try to project void faces as inner wires onto outer faces.
/// Returns (faces with added inner_wire UVs, remaining void faces that couldn't be projected).
#[allow(dead_code)] // Phase 3 integration into mesh_solid_with_voids
fn project_voids_as_inner_wires(
    store: &BRepStore,
    outer_face_keys: &[FaceKey],
    void_face_keys: &[FaceKey],
    tolerance: Real,
) -> (HashMap<FaceKey, Vec<Vec<(Real, Real)>>>, Vec<FaceKey>) {
    let mut face_inner_wires: HashMap<FaceKey, Vec<Vec<(Real, Real)>>> = HashMap::new();
    let mut remaining_voids: Vec<FaceKey> = Vec::new();

    for &vfk in void_face_keys {
        let mut projected = false;
        for &ofk in outer_face_keys {
            if let Some(inner_uvs) = project_void_face_as_inner_wire(store, ofk, vfk, tolerance) {
                face_inner_wires.entry(ofk).or_default().push(inner_uvs);
                projected = true;
                break; // Void face assigned to first matching outer face.
            }
        }
        if !projected {
            remaining_voids.push(vfk);
        }
    }

    (face_inner_wires, remaining_voids)
}

/// Tessellate a solid: outer shell + reversed void shells merged into one mesh.
pub fn mesh_solid_with_voids(
    store: &BRepStore,
    sk: SolidKey,
    config: &BRepMeshConfig,
    skip_faces: &[FaceKey],
) -> Option<SolidMeshOutput> {
    let solid = store.solids.get(sk)?;
    let outer_out = mesh_brep_shell_with_report(solid.outer_shell, store, config, skip_faces);
    if outer_out.mesh.vertices.is_empty() || outer_out.mesh.indices.is_empty() {
        return None;
    }

    let mut mesh = outer_out.mesh;
    let mut face_tri_ranges = face_ranges_from_report(&outer_out.report, 0);
    let mut report = outer_out.report;

    for &void_sk in &solid.void_shells {
        let void_out = mesh_brep_shell_with_report(void_sk, store, config, skip_faces);
        if void_out.mesh.vertices.is_empty() || void_out.mesh.indices.is_empty() {
            continue;
        }

        let tri_offset = mesh.indices.len() / 4;
        face_tri_ranges.extend(face_ranges_from_report(&void_out.report, tri_offset));

        let mut void_mesh = void_out.mesh;
        void_mesh.reverse_winding();
        mesh.append_from(&void_mesh);
        merge_shell_report(&mut report, &void_out.report);
    }
    // Weld once after all void shells are merged (avoids O(V²) per void)
    if !solid.void_shells.is_empty() {
        mesh.weld_vertices(config.weld_tolerance);
    }

    Some(SolidMeshOutput {
        mesh,
        face_tri_ranges,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::geom::curve2d::Curve2d;
    use crate::topo::{BRepFace, BRepShell, BRepWire, Orientation, WireKey};
    use rc3d_core::math::PVec3;

    fn build_plane_face(reg: &mut BRepStore, origin: PVec3, normal: PVec3, size: Real) -> FaceKey {
        let surface = SurfaceGeom::Plane { origin, normal, u_dir: PVec3::X };
        let v0 = reg.find_or_add_vertex(PVec3::new(origin.x, origin.y, origin.z), 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::new(origin.x + size, origin.y, origin.z), 1e-4);
        let v2 = reg.find_or_add_vertex(PVec3::new(origin.x + size, origin.y + size, origin.z), 1e-4);
        let v3 = reg.find_or_add_vertex(PVec3::new(origin.x, origin.y + size, origin.z), 1e-4);
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface: surface.clone(), outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![],
            color: None, degenerated_edges: vec![],
        });
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc.clone(), true);
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc.clone(), true);
        let e3 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, fk, pc.clone(), true);
        let e4 = reg.add_edge_with_pcurve(v3, v0, line.clone(), 1e-4, fk, pc, true);
        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, Orientation::Forward), (e2, Orientation::Forward),
            (e3, Orientation::Forward), (e4, Orientation::Forward),
        ];
        fk
    }

    #[test]
    fn test_project_void_as_inner_wire_plane() {
        let mut reg = BRepStore::new();
        let _outer = build_plane_face(&mut reg, PVec3::ZERO, PVec3::Z, 2.0);
        let void_fk = build_plane_face(&mut reg, PVec3::new(0.5, 0.5, 0.0), PVec3::Z, 0.5);
        let inner_uvs = project_void_face_as_inner_wire(&reg, _outer, void_fk, 1e-4);
        assert!(inner_uvs.is_some(), "co-planar void should project as inner wire");
        let uvs = inner_uvs.unwrap();
        assert!(uvs.len() >= 4, "expected >= 4 UV points, got {}", uvs.len());
        // Plane UVs are in world units: void at origin (0.5, 0.5), size 0.5.
        // UVs should be approximately (0.5±0.25, 0.5±0.25).
        for &(u, v) in &uvs[..4] {
            assert!((u - 0.5).abs() <= 0.6, "u={u:.3} expected near 0.5");
            assert!((v - 0.5).abs() <= 0.6, "v={v:.3} expected near 0.5");
        }
    }

    #[test]
    fn test_project_void_non_planar_fails() {
        let mut reg = BRepStore::new();
        let _outer = build_plane_face(&mut reg, PVec3::ZERO, PVec3::Z, 2.0);
        // Create a sphere face as void — can't project onto plane.
        let sphere = SurfaceGeom::Sphere { center: PVec3::ZERO, radius: 0.5 };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let void_fk = reg.faces.insert(BRepFace {
            surface: sphere, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![],
            color: None, degenerated_edges: vec![],
        });
        let inner_uvs = project_void_face_as_inner_wire(&reg, _outer, void_fk, 1e-4);
        assert!(inner_uvs.is_none(), "sphere-to-plane projection should fail");
    }

    #[test]
    fn merge_appends_void_tris_with_reversed_winding() {
        let outer_mesh = MeshResult {
            vertices: vec![
                PVec3::new(0.0, 0.0, 0.0),
                PVec3::new(1.0, 0.0, 0.0),
                PVec3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![PVec3::Z; 3],
            indices: vec![0, 1, 2, -1],
        };
        let void_mesh = MeshResult {
            vertices: vec![
                PVec3::new(0.0, 0.0, 1.0),
                PVec3::new(1.0, 0.0, 1.0),
                PVec3::new(0.0, 1.0, 1.0),
            ],
            normals: vec![PVec3::Z; 3],
            indices: vec![0, 1, 2, -1],
        };

        let mut combined = outer_mesh;
        let mut void_copy = void_mesh;
        void_copy.reverse_winding();
        combined.append_from(&void_copy);

        assert_eq!(combined.indices.len(), 8);
        assert_eq!(combined.indices[4..7], [3, 5, 4], "void tri winding reversed");
    }
}
