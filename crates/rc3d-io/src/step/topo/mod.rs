pub mod vertex;
pub mod edge;
pub mod shape;
pub mod build;

pub use vertex::{VertexId, TopoVertex, VertexRegistry};
pub use edge::{EdgeId, TopoEdge, EdgeRegistry, EdgeSense};
pub use shape::{TopoFace, TopoLoop, TopoShell, ShapeId};
pub use build::{TopoBuildResult, build_shared_topology};
