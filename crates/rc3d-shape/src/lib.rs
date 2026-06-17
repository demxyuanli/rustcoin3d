//! # rc3d-shape — B-Rep Shape Kernel
//!
//! Engine-level boundary representation kernel aligned with OpenCASCADE TopoDS + XDE.
//! Provides the complete B-Rep lifecycle: topology, geometry, healing, boolean operations,
//! tessellation, and XDE (Extended Data Exchange) layer.
//!
//! ## Architecture
//! - `topo` — topology graph (`FaceKey`, `EdgeKey`, `ShellKey`, `SolidKey` via slot maps)
//! - `geom` — curve and surface evaluation kernel (`CurveGeom`, `SurfaceGeom`)
//! - `store` — unified B-Rep store (`BRepStore`) holding topology + geometry + pcurves
//! - `shape` — public shape handle types (`Shape`, `ShapeId`, `ShapeKind`)
//! - `heal` — diagnostic and repair pipeline (OCC ShapeHealing equivalent)
//! - `bool` — boolean operations: union, intersection, difference (OCC BOPAlgo equivalent)
//! - `mesh` — B-Rep tessellation (OCC BRepMesh equivalent)
//! - `document` — shape document with metadata, PMI, and provenance tracking
//! - `xde` — extended data exchange: labels, attributes, color/material assignment
//! - `nurbs` — NURBS curve/surface support
//! - `tolerance` — configurable modeling tolerance context
//!
//! ## Key re-exports
//! - `Shape`, `ShapeId`, `ShapeKind`, `ShapeNode`, `ShapeType` — public shape handles
//! - `BRepStore`, `PCurveEdit`, `SameParamResult` — store and pcurve editing
//! - `ShapeDocument`, `PmiDataSet`, `PmiEntry`, `ProvenanceMap` — document layer
//! - `auto_heal_shell`, `HealLevel`, `HealReport` — healing pipeline entry point
//! - `SceneEmitPlan`, `EmitPlanOptions`, `EmitInstance` — mesh-to-scene emission
//! - `TessEntry`, `TessKey`, `TessellationCache` — tessellation cache
//!
//! ## OCC alignment
//! Corresponds to the OpenCASCADE ModelingData module: `TopoDS`, `Geom`, `BRep`,
//! `BRepTools`, `ShapeHealing`, `BOPAlgo`, `BRepMesh`, `XCAFDoc`.
//!
//! ## Usage
//! ```ignore
//! use rc3d_shape::{BRepStore, Shape, ShapeDocument};
//! use rc3d_shape::heal::{auto_heal_shell, HealLevel};
//! ```

pub mod bool;
pub mod brep;
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
pub use store::{BRepStore, PCurveEdit, SameParamResult};
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
