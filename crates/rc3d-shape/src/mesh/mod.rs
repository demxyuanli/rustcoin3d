//! # B-Rep Meshing Pipeline
//!
//! Tessellation of B-Rep faces and shells into triangle meshes.
//! OCC BRepMesh equivalent — incremental mesh with configurable deflection.
//!
//! ## Architecture
//! - **Face meshing** — per-face algorithm dispatch via `face_dispatch`
//!   - `face_fill` — Delaunay-based face filling (`BRepMesh_FastDiscretFace`)
//!   - `fill_surface` — UV-gridded surface fill with deflection control
//!   - `fill_plane` — analytic plane fill (no surface evaluation needed)
//!   - `fill_revolution` — polar-gridded revolution surface fill
//!   - `face_cdt` — constrained Delaunay triangulation of planar face domains
//!   - `delaunay2d` — generic 2D Delaunay triangulator
//! - **Shell meshing** — `shell_mesh::mesh_brep_shell()` aggregates per-face results
//! - **Solid meshing** — `solid_mesh::mesh_solid_with_voids()` handles void subtraction
//! - **Post-processing** — `post_process` handles gap repair, T-junction fixing, normal recompute
//!
//! ## Key types
//! | Module | Type/Function | OCC class |
//! |--------|--------------|-----------|
//! | `config` | `BRepMeshConfig`, `TessellationTier` | `BRepMesh_IncrementalMesh` |
//! | `shell_mesh` | `mesh_brep_shell()` | `BRepMesh_IncrementalMesh::Perform` |
//! | `face_dispatch` | `FaceMeshPlan`, `plan_face_mesh()` | Mesh selector |
//! | `report` | `ShellMeshReport`, `deflection_from_report()` | Mesh quality |
//! | `t4_quality` | `DeflectionMetrics`, `HausdorffMetrics` | Mesh validation |
//! | `edge_disc` | Edge discretization | `BRepMesh_Edge` |
//! | `post_process` | `heal_mesh_gaps()`, `fix_t_junctions()` | `BRepMesh_ModelHealer` |
//! | `same_param` | Same-parameter mesh stitch | `BRepMesh_ShapeTool` |
//!
//! ## Usage
//! ```ignore
//! use rc3d_shape::mesh::{mesh_brep_shell, BRepMeshConfig, TessellationTier};
//! let config = BRepMeshConfig::for_tier(TessellationTier::Standard);
//! let mesh = mesh_brep_shell(shell_key, &store, &config, &[]);
//! ```

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
pub mod post_process;
pub mod model_preprocessor;
mod ruled;
mod shell_pipeline;
mod shell_impl;

pub use config::{BRepMeshConfig, FallbackAllowlist, TessellationPolicy, TessellationTier, MESH_CLOSED_SURFACE_SEGS};
pub use edge_pool::{measure_equivalent_edge_weld_gap, measure_face_boundary_surface_gap};
pub use solid_mesh::mesh_solid_with_voids;
pub use face_dispatch::{algo_from_plan, FaceMeshPlan, SurfaceFillReason, plan_face_mesh};
pub use shell_mesh::{mesh_brep_shell, mesh_brep_shell_with_report, ShellMeshOutput};
pub(crate) use param_div::curvature_driven_divisions;

#[cfg(test)]
mod mesh_integration {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use rc3d_core::math::PVec3;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::store::BRepStore;
    use crate::topo::{BRepFace, BRepShell, BRepWire, BRepSolid, Orientation, ShellKey};

    fn build_plane_square_shell() -> (BRepStore, ShellKey) {
        let mut reg = BRepStore::new();
        let face_key = {
            let wire = reg.wires.insert(BRepWire { edges: vec![] });
            reg.faces.insert(BRepFace {
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
            })
        };
        let edges_data = [
            (PVec3::ZERO, PVec3::new(10.0, 0.0, 0.0), (0.0, 0.0), (10.0, 0.0)),
            (PVec3::new(10.0, 0.0, 0.0), PVec3::new(10.0, 10.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
            (PVec3::new(10.0, 10.0, 0.0), PVec3::new(0.0, 10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
            (PVec3::new(0.0, 10.0, 0.0), PVec3::ZERO, (0.0, 10.0), (0.0, 0.0)),
        ];
        let mut wire_edges = Vec::new();
        for (a, b, u0, u1) in edges_data {
            let v0 = reg.find_or_add_vertex(a, 1e-4);
            let v1 = reg.find_or_add_vertex(b, 1e-4);
            let curve_3d = CurveGeom::Line { origin: a, direction: b - a };
            let pcurve = Curve2d::Line { origin: (u0.0, u0.1), direction: (u1.0 - u0.0, u1.1 - u0.1) };
            let ek = reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, pcurve, true);
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
        let (mut reg, shell_key) = build_plane_square_shell();
        let out = mesh_brep_shell_with_report(shell_key, &mut reg, &BRepMeshConfig::default(), &[]);
        let tri_count = out.mesh.indices.len() / 4;
        assert!(tri_count >= 2, "expected interior fill, got {tri_count} tris");
        assert!(!out.mesh.vertices.is_empty());
        assert!(out.report.meshed_faces >= 1);
    }
}
