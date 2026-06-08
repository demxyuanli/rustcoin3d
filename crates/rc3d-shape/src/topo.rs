//! B-Rep topology types. T1.3-T1.4

use std::collections::HashMap;
use slotmap::new_key_type;
use rc3d_core::math::Vec3;
use crate::geom::CurveGeom;
use crate::geom::curve2d::Curve2d;

new_key_type! { pub struct VertexKey; }
new_key_type! { pub struct EdgeKey; }
new_key_type! { pub struct WireKey; }
new_key_type! { pub struct FaceKey; }
new_key_type! { pub struct ShellKey; }
new_key_type! { pub struct SolidKey; }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation { Forward, Reversed, Internal, External }

#[derive(Debug, Clone)]
pub struct BRepVertex { pub position: Vec3, pub tolerance: f32 }

#[derive(Debug, Clone)]
pub struct BRepEdge {
    pub curve: CurveGeom,
    pub tolerance: f32,
    /// Canonical low vertex (min SlotMap key) — mesh t=0
    pub v_low: VertexKey,
    /// Canonical high vertex (max SlotMap key) — mesh t=1
    pub v_high: VertexKey,
    /// PCurves: 2D curves in UV space of each referencing face.
    /// OCC alignment: BRep_CurveOnSurface (Geom2d_Curve per face).
    pub pcurves: HashMap<FaceKey, Curve2d>,
}

#[derive(Debug, Clone)]
pub struct BRepWire { pub edges: Vec<(EdgeKey, Orientation)> }

#[derive(Debug, Clone)]
pub struct BRepFace {
    pub surface: crate::geom::SurfaceGeom,
    pub outer_wire: WireKey,
    pub inner_wires: Vec<WireKey>,
    pub same_sense: bool,
    pub tolerance: f32,
    /// Seam edges on closed parametric faces (OCCT ShapeFix_Face::FixMissingSeamMode).
    pub seam_edges: Vec<EdgeKey>,
    /// Surface color from STYLED_ITEM, if present in the STEP file.
    pub color: Option<[f32; 3]>,
    /// Degenerated edges at surface singularities (Phase 2+).
    pub degenerated_edges: Vec<EdgeKey>,
}

#[derive(Debug, Clone)]
pub struct BRepShell { pub faces: Vec<(FaceKey, Orientation)>, pub closed: bool, pub step_id: Option<u64> }

#[derive(Debug, Clone)]
pub struct BRepSolid { pub outer_shell: ShellKey, pub void_shells: Vec<ShellKey> }
