pub mod edge_disc;
pub(crate) mod refiner;
pub(crate) mod optimize;
pub mod face_uv;
pub mod face_fill;
pub mod fill_surface;
pub mod fill_plane;
pub mod fill_revolution;
pub mod face_cdt;
pub mod delaunay2d;
pub(crate) mod same_param;
pub mod report;
pub mod t4_quality;
pub(crate) mod diagnostic;
pub(crate) mod algo_factory;
pub(crate) mod void_subtract;
pub mod face_dispatch;
pub mod shell_mesh;
pub(crate) mod orient;
pub(crate) mod uv_source;
pub(crate) mod solid_mesh;

mod boundary;
pub mod config;
mod fallback_policy;
mod grid;
mod edge_pool;
mod param_div;
mod post_process;
mod ruled;
mod shell_pipeline;
mod shell_impl;

pub use config::{BRepMeshConfig, MESH_CLOSED_SURFACE_SEGS};
pub use edge_pool::{measure_equivalent_edge_weld_gap, measure_face_boundary_surface_gap};
pub use solid_mesh::mesh_solid_with_voids;
pub use face_dispatch::{algo_from_plan, FaceMeshPlan, SurfaceFillReason, plan_face_mesh};
pub use shell_mesh::{mesh_brep_shell, mesh_brep_shell_with_report, ShellMeshOutput};
pub(crate) use param_div::curvature_driven_divisions;

#[cfg(test)]
mod mesh_integration {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use rc3d_core::math::Vec3;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::store::BRepStore;
    use crate::topo::{BRepFace, BRepShell, BRepWire, BRepSolid, Orientation, ShellKey};

    fn build_plane_square_shell() -> (BRepStore, ShellKey) {
        let mut reg = BRepStore::new();
        let face_key = {
            let wire = reg.wires.insert(BRepWire { edges: vec![] });
            reg.faces.insert(BRepFace {
                surface: SurfaceGeom::Plane {
                    origin: Vec3::ZERO,
                    normal: Vec3::Z,
                    u_dir: Vec3::X,
                },
                outer_wire: wire,
                inner_wires: vec![],
                same_sense: true,
                tolerance: 1e-4,
                seam_edges: vec![],
                color: None,
            degenerated_edges: vec![],
            })
        };
        let edges_data = [
            (Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), (0.0, 0.0), (10.0, 0.0)),
            (Vec3::new(10.0, 0.0, 0.0), Vec3::new(10.0, 10.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
            (Vec3::new(10.0, 10.0, 0.0), Vec3::new(0.0, 10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
            (Vec3::new(0.0, 10.0, 0.0), Vec3::ZERO, (0.0, 10.0), (0.0, 0.0)),
        ];
        let mut wire_edges = Vec::new();
        for (a, b, u0, u1) in edges_data {
            let v0 = reg.find_or_add_vertex(a, 1e-4);
            let v1 = reg.find_or_add_vertex(b, 1e-4);
            let curve_3d = CurveGeom::Line { origin: a, direction: b - a };
            let pcurve = Curve2d::Line { origin: (u0.0, u0.1), direction: (u1.0 - u0.0, u1.1 - u0.1) };
            let ek = reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, (pcurve, true));
            wire_edges.push((ek, Orientation::Forward));
        }
        let outer_wire = reg.wires.insert(BRepWire { edges: wire_edges });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = outer_wire;
        }
        let shell_key = reg.shells.insert(BRepShell {
            faces: vec![(face_key, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let _ = reg.solids.insert(BRepSolid {
            outer_shell: shell_key,
            void_shells: vec![],
        });
        (reg, shell_key)
    }

    #[test]
    fn mesh_brep_shell_plane_face_has_interior_tris() {
        let (reg, shell_key) = build_plane_square_shell();
        let out = mesh_brep_shell_with_report(shell_key, &reg, &BRepMeshConfig::default(), &[]);
        let tri_count = out.mesh.indices.len() / 4;
        assert!(tri_count >= 2, "expected interior fill, got {tri_count} tris");
        assert!(!out.mesh.vertices.is_empty());
        assert!(out.report.meshed_faces >= 1);
    }
}