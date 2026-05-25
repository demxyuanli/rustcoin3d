pub mod geom;
pub mod topo;
pub mod registry;
pub mod build;
pub mod overlay;
pub mod heal;
pub mod mesh;

pub use geom::{CurveGeom, SurfaceGeom};
pub use topo::{Orientation, BRepVertex, BRepEdge, BRepWire, BRepFace, BRepShell, BRepSolid};
pub use registry::BRepRegistry;
pub use build::{BRepBuildResult, build_brep};
