pub mod vertex;
pub mod edge;
pub mod shape;

pub use vertex::{VertexId, TopoVertex, VertexRegistry};
pub use edge::{EdgeId, TopoEdge, EdgeRegistry, EdgeSense};
pub use shape::{TopoFace, TopoLoop, TopoShell, ShapeId};
