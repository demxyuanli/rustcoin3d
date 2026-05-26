//! B-Rep topology types. T1.3-T1.4

use std::collections::HashMap;
use slotmap::new_key_type;
use rc3d_core::math::Vec3;
use super::geom::CurveGeom;

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
    pub pcurves: HashMap<FaceKey, CurveGeom>,
}

#[derive(Debug, Clone)]
pub struct BRepWire { pub edges: Vec<(EdgeKey, Orientation)> }

#[derive(Debug, Clone)]
pub struct BRepFace {
    pub surface: super::geom::SurfaceGeom,
    pub outer_wire: WireKey,
    pub inner_wires: Vec<WireKey>,
    pub same_sense: bool,
    pub tolerance: f32,
    /// Seam edges on closed parametric faces (OCCT ShapeFix_Face::FixMissingSeamMode).
    pub seam_edges: Vec<EdgeKey>,
    /// Surface color from STYLED_ITEM, if present in the STEP file.
    pub color: Option<[f32; 3]>,
}

#[derive(Debug, Clone)]
pub struct BRepShell { pub faces: Vec<(FaceKey, Orientation)>, pub closed: bool, pub step_id: Option<u64> }

#[derive(Debug, Clone)]
pub struct BRepSolid { pub outer_shell: ShellKey, pub void_shells: Vec<ShellKey> }
