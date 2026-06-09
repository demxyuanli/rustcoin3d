//! Engine-level B-Rep shape document kernel (OCC TopoDS + XDE aligned).

pub mod bool;
pub mod document;
pub mod emit_plan;
pub mod error;
pub mod geom;
pub mod heal;
pub mod mesh;
pub mod mesh_result;
pub mod mesh_split;
pub mod nurbs;
pub mod shape;
pub mod store;
pub mod tolerance;
pub mod tessellation;
pub mod topo;
pub mod topo_iter;
pub mod xde;

pub use document::{PmiDataSet, PmiEntry, ProvenanceMap, ShapeDocument};
pub use emit_plan::{
    CachedMesh, EmitInstance, EmitNode, EmitPlanOptions, FaceMaterialGroup, MaterialDesc,
    MeshSlotId, PmiPlacement, SceneEmitPlan,
};
pub use error::ShapeError;
pub use mesh_result::MeshResult;
pub use shape::{Shape, ShapeId, ShapeKind, ShapeNode, ShapeType};
pub use store::{BRepStore, PCurveEdit};
pub use tolerance::{
    ToleranceContext, DEFAULT_MODEL_TOLERANCE, MAX_MODEL_TOLERANCE, TOLERANCE_FLOOR,
};
pub use mesh_split::{extract_mesh_tri_ranges, face_material_draw_batches, FaceDrawBatch, FaceTriRange};
pub use tessellation::{TessEntry, TessKey, TessellationCache};
pub use topo::*;
pub use xde::{AttributeBag, LabelId, XdeLabel, XdeLabelForest};

pub use heal::topo_diag::{check_shell_topo_diag, TopoDiagReport};
pub use heal::{auto_heal_shell, check_shell_continuity, HealLevel, HealReport};
pub use mesh::{measure_equivalent_edge_weld_gap, measure_face_boundary_surface_gap};
pub use mesh::report::deflection_from_report;
pub use mesh::t4_quality::{
    deflection_within_band, hausdorff_meshes, measure_shell_deflection, DeflectionMetrics,
    HausdorffMetrics,
};
