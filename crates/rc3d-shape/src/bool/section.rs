//! Boolean section operation: extract intersection curves as wireframe.
//!
//! Runs Phase 1 (PaveFiller) only, without face splitting, classification,
//! or stitching. Returns the raw 3D intersection curves as polylines.
//!
//! OCC alignment: BOPAlgo_Section — intersection wire extraction without
//! full boolean solid computation.

use rc3d_core::math::Real;
use crate::store::BRepStore;
use crate::topo::ShellKey;
use crate::geom::CurveGeom;

/// Result of a section (wireframe intersection) operation.
pub struct SectionResult {
    /// 3D intersection curves (polylines from marching/analytic SSI).
    pub curves: Vec<CurveGeom>,
    /// Number of face pairs that produced intersection curves.
    pub face_pairs: usize,
    /// Whether any intersection was found.
    pub is_empty: bool,
}

/// Compute the section (intersection wireframe) between two sets of shells.
///
/// Equivalent to OCC's `BOPAlgo_Section::Perform()`. Runs only Phase 1
/// (face-face intersection via PaveFiller), extracting the computed 3D
/// intersection curves without splitting faces or classifying regions.
pub fn boolean_section(
    shells_a: &[ShellKey],
    shells_b: &[ShellKey],
    reg: &BRepStore,
    tolerance: Real,
) -> SectionResult {
    let mut result = SectionResult {
        curves: Vec::new(),
        face_pairs: 0,
        is_empty: true,
    };

    if shells_a.is_empty() || shells_b.is_empty() {
        return result;
    }

    // Run PaveFiller: Phase 1 only
    let (bopds, _pave_report) = super::pave_filler::fill_paves(shells_a, shells_b, reg, tolerance);

    for interf in &bopds.face_face_interfs {
        if interf.curves_3d.is_empty() {
            continue;
        }
        result.face_pairs += 1;
        for curve in &interf.curves_3d {
            result.curves.push(curve.clone());
        }
    }

    result.is_empty = result.curves.is_empty();
    result
}

/// Compute the section between two shells and return the intersection
/// curves as a single composite polyline (for visualization/debugging).
pub fn section_as_polyline(
    shells_a: &[ShellKey],
    shells_b: &[ShellKey],
    reg: &BRepStore,
    tolerance: Real,
) -> Option<CurveGeom> {
    let section = boolean_section(shells_a, shells_b, reg, tolerance);
    if section.curves.is_empty() {
        return None;
    }
    // Collect all sampled points from intersection curves
    let mut all_pts = Vec::new();
    for curve in &section.curves {
        let n = 32usize;
        for i in 0..=n {
            let t = i as Real / n as Real;
            all_pts.push(curve.d0(t));
        }
    }
    if all_pts.len() >= 2 {
        Some(CurveGeom::Polyline { points: all_pts })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::*;
    use crate::store::BRepStore;
    use rc3d_core::math::PVec3;

    fn make_plane_shell(reg: &mut BRepStore, origin: PVec3, normal: PVec3) -> ShellKey {
        let surface = SurfaceGeom::Plane {
            origin,
            normal,
            u_dir: if normal.z.abs() < 0.9 { PVec3::Z } else { PVec3::X },
        };
        let v0 = reg.find_or_add_vertex(origin + PVec3::new(-10.0, -10.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(origin + PVec3::new(10.0, -10.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(origin + PVec3::new(10.0, 10.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(origin + PVec3::new(-10.0, 10.0, 0.0), 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface: surface.clone(), outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![],
            color: None, degenerated_edges: vec![],
        });
        let corners = [(v0,v1), (v1,v2), (v2,v3), (v3,v0)];
        let mut wire_edges = Vec::new();
        for (va, vb) in corners {
            let pa = reg.vertices.get(va).unwrap().position;
            let pb = reg.vertices.get(vb).unwrap().position;
            let line = CurveGeom::Line { origin: pa, direction: pb - pa };
            let pc = crate::geom::curve2d::Curve2d::Line {
                origin: (pa.x, pa.y), direction: (pb.x - pa.x, pb.y - pa.y),
            };
            let ek = reg.add_edge_with_pcurve(va, vb, line, 1e-4, fk, pc, true);
            wire_edges.push((ek, Orientation::Forward));
        }
        reg.wires.get_mut(wk).unwrap().edges = wire_edges;
        reg.shells.insert(BRepShell { faces: vec![(fk, Orientation::Forward)], closed: false, step_id: None })
    }

    #[test]
    fn test_section_intersecting_planes() {
        let mut reg = BRepStore::new();
        let sa = make_plane_shell(&mut reg, PVec3::ZERO, PVec3::Z);
        let sb = make_plane_shell(&mut reg, PVec3::ZERO, PVec3::Y);
        let result = boolean_section(&[sa], &[sb], &reg, 1e-4);
        assert!(!result.is_empty, "intersecting planes should produce section");
        assert!(result.face_pairs > 0);
        assert!(!result.curves.is_empty());
    }

    #[test]
    fn test_section_disjoint() {
        let mut reg = BRepStore::new();
        let sa = make_plane_shell(&mut reg, PVec3::new(0.0, 0.0, 0.0), PVec3::Z);
        let sb = make_plane_shell(&mut reg, PVec3::new(0.0, 0.0, 100.0), PVec3::Z);
        let result = boolean_section(&[sa], &[sb], &reg, 1e-4);
        assert!(result.is_empty);
    }

    #[test]
    fn test_section_as_polyline() {
        let mut reg = BRepStore::new();
        let sa = make_plane_shell(&mut reg, PVec3::ZERO, PVec3::Z);
        let sb = make_plane_shell(&mut reg, PVec3::ZERO, PVec3::Y);
        let poly = section_as_polyline(&[sa], &[sb], &reg, 1e-4);
        assert!(poly.is_some(), "should produce intersection polyline");
    }
}
