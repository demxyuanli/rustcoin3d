pub mod geom;
pub mod topo;
pub mod registry;
pub mod build;
pub mod overlay;
pub mod heal;
pub mod mesh;

pub use overlay::build_edge_curves;
pub use geom::{CurveGeom, SurfaceGeom, SurfaceParamRange};
pub use topo::{Orientation, BRepVertex, BRepEdge, BRepWire, BRepFace, BRepShell, BRepSolid};
pub use registry::BRepRegistry;
pub use build::{
    build_brep, build_brep_with_options, BRepBuildOptions, BRepBuildReport, BRepBuildResult,
};
pub use mesh::report::deflection_from_report;
pub use mesh::t4_quality::{DeflectionMetrics, HausdorffMetrics, deflection_within_band, hausdorff_meshes, measure_shell_deflection};
