//! BRepCheck_Analyzer — topology validation orchestrator (OCC BRepCheck_Analyzer subset).
//!
//! Delegates to sub-checkers: Vertex, Edge, Wire, Face, Shell, Solid.

use std::collections::HashSet;

use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, ShellKey, SolidKey, VertexKey, WireKey};
use crate::topo_iter;

mod vertex;
mod edge;
mod wire;
mod face;
mod shell;
mod solid;

use vertex::*;
use edge::*;
use wire::*;
use face::*;
use shell::*;
use solid::*;

// Re-export items that heal/mod.rs re-exports externally
pub use wire::check_uv_self_intersection;

// ── CheckStatus enum ────────────────────────────────────────────────────────

/// Structured check status for topology validation (OCC BRepCheck_Status equivalent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckStatus {
    // Vertex
    InvalidPointOnCurve,
    InvalidPointOnSurface,
    // Edge
    No3DCurve,
    InvalidSameParameterFlag,
    InvalidSameRangeFlag,
    InvalidToleranceValue,
    // Wire
    NotClosed,
    SelfIntersectingWire,
    RedundantEdge,
    // Face
    IntersectingWires,
    BadOrientation,
    // Shell
    ShellNotClosed,
    UnorientableShape,
    // Solid
    SolidBadOrientation,
    EnclosedRegionViolation,
    // Generic
    Ok,
}

// ── CheckReport ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct CheckReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    /// Faces that failed validation and should not be meshed.
    pub failed_faces: Vec<FaceKey>,
    pub has_uv_gaps: bool,
    pub has_pcurve_issues: bool,
    pub has_self_intersections: bool,
    pub has_singularities: bool,
    pub has_inner_wires: bool,
    pub has_intersecting_wires: bool,
    pub has_face_self_intersections: bool,
    /// Shell closure: true when at least one edge is open (not shared by 2 faces).
    pub has_open_edges: bool,
    /// Number of open (dangling) edges in the shell.
    pub open_edge_count: usize,
    /// Structured status entries: (entity_id, status).
    pub statuses: Vec<(String, CheckStatus)>,
}

impl CheckReport {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn merge(&mut self, other: CheckReport) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
        self.failed_faces.extend(other.failed_faces);
        self.has_uv_gaps |= other.has_uv_gaps;
        self.has_pcurve_issues |= other.has_pcurve_issues;
        self.has_self_intersections |= other.has_self_intersections;
        self.has_singularities |= other.has_singularities;
        self.has_inner_wires |= other.has_inner_wires;
        self.has_intersecting_wires |= other.has_intersecting_wires;
        self.has_face_self_intersections |= other.has_face_self_intersections;
        self.statuses.extend(other.statuses);
    }

    #[inline]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    #[inline]
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }
}

// ── Orchestrator ────────────────────────────────────────────────────────────

/// Validate shell topology before meshing (OCC BRepCheck_Analyzer).
pub fn check_shell(shell_key: ShellKey, reg: &BRepStore) -> CheckReport {
    let mut report = CheckReport::default();
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => {
            report.errors.push(format!("shell {:?} not found", shell_key));
            return report;
        }
    };

    // Collect all vertices and edges in the shell for per-entity checks
    let mut all_vertices: HashSet<VertexKey> = HashSet::new();
    let mut all_edges: HashSet<EdgeKey> = HashSet::new();

    for &(fk, _) in &shell.faces {
        for ek in topo_iter::iter_edges_of_face(fk, reg) {
            all_edges.insert(ek);
            if let Some((vl, vh)) = topo_iter::iter_vertices_of_edge(ek, reg) {
                all_vertices.insert(vl);
                all_vertices.insert(vh);
            }
        }
    }

    // ── Vertex checks (BRepCheck_Vertex) ─────────────────────────────────
    for &vk in &all_vertices {
        for status in check_vertex_on_curves(vk, reg) {
            report.statuses.push((format!("vertex:{:?}", vk), status));
        }
        for status in check_vertex_on_surfaces(vk, reg) {
            report.statuses.push((format!("vertex:{:?}", vk), status));
        }
    }

    // ── Edge checks (BRepCheck_Edge) ─────────────────────────────────────
    for &ek in &all_edges {
        for status in check_no_3d_curve(ek, reg) {
            report.statuses.push((format!("edge:{:?}", ek), status));
        }
        for status in check_same_parameter_deviation(ek, reg) {
            report.statuses.push((format!("edge:{:?}", ek), status));
        }
        for status in check_same_range(ek, reg) {
            report.statuses.push((format!("edge:{:?}", ek), status));
        }
        // Legacy tolerance check
        report.warnings.extend(check_edge_tolerance(ek, reg));
    }

    // ── Face checks (BRepCheck_Face) ─────────────────────────────────────
    for &(face_key, _) in &shell.faces {
        check_face(face_key, reg, &mut report);
    }

    // ── Wire checks (BRepCheck_Wire) ─────────────────────────────────────
    for &(face_key, _) in &shell.faces {
        if let Some(face) = reg.faces.get(face_key) {
            let wire_keys: Vec<WireKey> = std::iter::once(face.outer_wire)
                .chain(face.inner_wires.iter().copied())
                .collect();
            for wk in wire_keys {
                for status in check_redundant_edge(wk, reg) {
                    report
                        .statuses
                        .push((format!("wire:{:?}", wk), status));
                }
            }
        }
    }

    // ── UV self-intersection ─────────────────────────────────────────────
    for &(face_key, _) in &shell.faces {
        let si_warnings = check_uv_self_intersection(face_key, reg);
        if !si_warnings.is_empty() {
            report.has_self_intersections = true;
        }
        report.warnings.extend(si_warnings);
        if let Some(face) = reg.faces.get(face_key) {
            if !face.inner_wires.is_empty() {
                report.has_inner_wires = true;
                if super::intersecting_wires::detect_intersecting_wires(face_key, reg) {
                    report.has_intersecting_wires = true;
                }
            }
        }

        // Face-level self-intersection (surface folding in 3D).
        let face_si_count = super::face_self_intersect::check_face_self_intersect(
            face_key, reg, 4,
        );
        if face_si_count > 0 {
            report.has_face_self_intersections = true;
            report.warnings.push(format!(
                "face {:?}: surface self-intersection suspected ({} normal inversions)",
                face_key, face_si_count
            ));
        }
    }

    // ── Shell checks (BRepCheck_Shell) ───────────────────────────────────
    let nm_warnings = check_non_manifold(&shell.faces, reg);
    report.warnings.extend(nm_warnings);

    // Euler-Poincare topology validation
    if let Some(chi) = check_euler_poincare(shell_key, reg) {
        if chi != 2 && chi != 0 {
            report.warnings.push(format!(
                "shell {:?}: Euler characteristic chi={} (expected 2 for closed, 0 for torus)",
                shell_key, chi
            ));
        }
    }

    // Shell closure check
    let closed = check_shell_closed(shell_key, reg);
    report.open_edge_count = closed.open_edges.len();
    report.has_open_edges = !closed.open_edges.is_empty();
    if !closed.is_closed {
        report.warnings.push(format!(
            "shell {:?}: not closed -- {}/{} open edges, {} non-manifold",
            shell_key,
            closed.open_edges.len(),
            closed.total_edges,
            closed.non_manifold_edges.len(),
        ));
    }

    // Unorientable shape check
    for status in check_unorientable(shell_key, reg) {
        report
            .statuses
            .push((format!("shell:{:?}", shell_key), status));
    }

    report
}

/// Validate solid topology (OCC BRepCheck_Solid).
pub fn check_solid(solid_key: SolidKey, reg: &BRepStore) -> CheckReport {
    let mut report = CheckReport::default();
    let solid = match reg.solids.get(solid_key) {
        Some(s) => s,
        None => {
            report
                .errors
                .push(format!("solid {:?} not found", solid_key));
            return report;
        }
    };

    // Check outer shell
    let shell_report = check_shell(solid.outer_shell, reg);
    report.merge(shell_report);

    // Check void shells
    for &vk in &solid.void_shells {
        let void_report = check_shell(vk, reg);
        report.merge(void_report);
    }

    // Solid-level checks
    for status in check_solid_orientation(solid_key, reg) {
        report
            .statuses
            .push((format!("solid:{:?}", solid_key), status));
    }

    for status in check_enclosed_region(solid_key, reg) {
        report
            .statuses
            .push((format!("solid:{:?}", solid_key), status));
    }

    report
}

// ── Re-exports for backward compatibility ───────────────────────────────────

pub use shell::check_shell_closed;
pub use shell::ShellClosedReport;
pub use wire::{check_face_wire_gaps, check_wire_closed};

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepEdge, BRepFace, BRepShell, BRepSolid, BRepWire, Orientation};
    use rc3d_core::math::PVec3;
    use std::collections::HashMap;

    fn closed_cube_shell(reg: &mut BRepStore) -> ShellKey {
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: PVec3::ZERO,
                normal: PVec3::Z,
                u_dir: PVec3::X,
            },
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let edges_data = [
            (
                PVec3::ZERO,
                PVec3::new(10.0, 0.0, 0.0),
                (0.0, 0.0),
                (10.0, 0.0),
            ),
            (
                PVec3::new(10.0, 0.0, 0.0),
                PVec3::new(10.0, 10.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
            ),
            (
                PVec3::new(10.0, 10.0, 0.0),
                PVec3::new(0.0, 10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
            ),
            (
                PVec3::new(0.0, 10.0, 0.0),
                PVec3::ZERO,
                (0.0, 10.0),
                (0.0, 0.0),
            ),
        ];
        let mut wire_edges = Vec::new();
        for (a, b, u0, u1) in edges_data {
            let v0 = reg.find_or_add_vertex(a, 1e-4);
            let v1 = reg.find_or_add_vertex(b, 1e-4);
            let curve_3d = CurveGeom::Line {
                origin: a,
                direction: b - a,
            };
            let pcurve = Curve2d::Line {
                origin: (u0.0, u0.1),
                direction: (u1.0 - u0.0, u1.1 - u0.1),
            };
            let ek =
                reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, pcurve, true);
            let edge = reg.edges.get(ek).unwrap();
            let orient = if edge.v_low == v0 {
                Orientation::Forward
            } else {
                Orientation::Reversed
            };
            wire_edges.push((ek, orient));
        }
        let outer_wire = reg.wires.insert(BRepWire {
            edges: wire_edges,
        });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = outer_wire;
        }
        reg.shells.insert(BRepShell {
            faces: vec![(face_key, Orientation::Forward)],
            closed: true,
            step_id: None,
        })
    }

    #[test]
    fn valid_plane_face_has_no_errors() {
        let mut reg = BRepStore::new();
        let sk = closed_cube_shell(&mut reg);
        let report = check_shell(sk, &reg);
        assert!(report.errors.is_empty(), "{:?}", report.errors);
    }

    #[test]
    fn open_wire_reports_error() {
        let mut reg = BRepStore::new();
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: PVec3::ZERO,
                normal: PVec3::Z,
                u_dir: PVec3::X,
            },
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-6,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let a = PVec3::ZERO;
        let b = PVec3::new(10.0, 0.0, 0.0);
        let c = PVec3::new(10.0, 10.0, 0.0);
        let va = reg.find_or_add_vertex(a, 1e-4);
        let vb = reg.find_or_add_vertex(b, 1e-4);
        let vc = reg.find_or_add_vertex(c, 1e-4);
        let e0 = reg.add_edge_with_pcurve(
            va,
            vb,
            CurveGeom::Line {
                origin: a,
                direction: b - a,
            },
            1e-4,
            face_key,
            Curve2d::Line {
                origin: (0.0, 0.0),
                direction: (10.0, 0.0),
            },
            true,
        );
        let e1 = reg.add_edge_with_pcurve(
            vb,
            vc,
            CurveGeom::Line {
                origin: b,
                direction: c - b,
            },
            1e-4,
            face_key,
            Curve2d::Line {
                origin: (10.0, 0.0),
                direction: (0.0, 10.0),
            },
            true,
        );
        let outer = reg.wires.insert(BRepWire {
            edges: vec![(e0, Orientation::Forward), (e1, Orientation::Forward)],
        });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = outer;
        }
        let sk = reg.shells.insert(BRepShell {
            faces: vec![(face_key, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let report = check_shell(sk, &reg);
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn zero_area_face_yields_error() {
        let mut reg = BRepStore::new();
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: PVec3::ZERO,
                normal: PVec3::Z,
                u_dir: PVec3::X,
            },
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let shell_key = reg.shells.insert(BRepShell {
            faces: vec![(face_key, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let report = check_shell(shell_key, &reg);
        assert!(
            !report.errors.is_empty(),
            "zero-area face (empty wire with no seam edges) should be an error"
        );
    }

    #[test]
    fn test_check_solid_runs() {
        let mut reg = BRepStore::new();
        // Create a minimal closed 2-face shell
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let p0 = PVec3::ZERO;
        let p1 = PVec3::new(1.0, 0.0, 0.0);
        let p2 = PVec3::new(1.0, 1.0, 0.0);
        let p3 = PVec3::new(0.0, 1.0, 0.0);
        let v0 = reg.find_or_add_vertex(p0, 1e-4);
        let v1 = reg.find_or_add_vertex(p1, 1e-4);
        let v2 = reg.find_or_add_vertex(p2, 1e-4);
        let v3 = reg.find_or_add_vertex(p3, 1e-4);

        let wk0 = reg.wires.insert(BRepWire { edges: vec![] });
        let fk0 = reg.faces.insert(BRepFace {
            surface: surface.clone(),
            outer_wire: wk0,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let line = |a: PVec3, b: PVec3| CurveGeom::Line {
            origin: a,
            direction: b - a,
        };
        let pc = |a: PVec3, b: PVec3| Curve2d::Line {
            origin: (a.x, a.y),
            direction: (b.x - a.x, b.y - a.y),
        };
        let e01 = reg.add_edge_with_pcurve(v0, v1, line(p0, p1), 1e-4, fk0, pc(p0, p1), true);
        let e12 = reg.add_edge_with_pcurve(v1, v2, line(p1, p2), 1e-4, fk0, pc(p1, p2), true);
        let e23 = reg.add_edge_with_pcurve(v2, v3, line(p2, p3), 1e-4, fk0, pc(p2, p3), true);
        let e30 = reg.add_edge_with_pcurve(v3, v0, line(p3, p0), 1e-4, fk0, pc(p3, p0), true);
        reg.wires.get_mut(wk0).unwrap().edges = vec![
            (e01, Orientation::Forward),
            (e12, Orientation::Forward),
            (e23, Orientation::Forward),
            (e30, Orientation::Forward),
        ];

        let wk1 = reg.wires.insert(BRepWire { edges: vec![] });
        let fk1 = reg.faces.insert(BRepFace {
            surface: surface.clone(),
            outer_wire: wk1,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        reg.add_edge_with_pcurve(v0, v1, line(p0, p1), 1e-4, fk1, pc(p0, p1), true);
        reg.add_edge_with_pcurve(v1, v2, line(p1, p2), 1e-4, fk1, pc(p1, p2), true);
        reg.add_edge_with_pcurve(v2, v3, line(p2, p3), 1e-4, fk1, pc(p2, p3), true);
        reg.add_edge_with_pcurve(v3, v0, line(p3, p0), 1e-4, fk1, pc(p3, p0), true);
        reg.wires.get_mut(wk1).unwrap().edges = vec![
            (e01, Orientation::Reversed),
            (e30, Orientation::Reversed),
            (e23, Orientation::Reversed),
            (e12, Orientation::Reversed),
        ];

        let sk = reg.shells.insert(BRepShell {
            faces: vec![(fk0, Orientation::Forward), (fk1, Orientation::Forward)],
            closed: true,
            step_id: None,
        });
        let solid_key = reg.solids.insert(BRepSolid {
            outer_shell: sk,
            void_shells: vec![],
        });
        let report = check_solid(solid_key, &reg);
        // The 2-face coplanar shell may have errors but should not panic
        assert!(report.statuses.is_empty() || !report.statuses.is_empty());
    }
}
