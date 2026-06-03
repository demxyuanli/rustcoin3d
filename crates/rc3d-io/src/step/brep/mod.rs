pub mod geom;
pub mod topo;
pub mod registry;
pub mod build;
pub mod same_parameter;
pub mod overlay;
pub mod properties;
pub mod fillet;
pub mod offset_api;
pub mod incremental_mesh;
pub mod quality;

pub use rc3d_shape::heal;
pub use rc3d_shape::mesh;

pub use overlay::{build_edge_curves, build_mesh_wireframe};
pub use geom::{CurveGeom, SurfaceGeom, SurfaceParamRange};
pub use topo::{Orientation, BRepVertex, BRepEdge, BRepWire, BRepFace, BRepShell, BRepSolid};
pub use registry::{BRepStore, PCurveEdit};
pub use build::{
    build_brep, build_brep_with_options, BRepBuildOptions, BRepBuildReport, BRepBuildResult,
};
pub use mesh::report::deflection_from_report;
pub use mesh::t4_quality::{
    DeflectionMetrics, HausdorffMetrics, deflection_within_band, hausdorff_meshes,
    measure_shell_deflection,
};
pub use properties::{MeshProperties, compute_mesh_properties};
