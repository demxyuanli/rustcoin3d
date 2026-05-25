//! Edge curve overlay for scene graph (moved from mod.rs).

use rc3d_scene::SceneGraph;
use super::topo::SolidKey;
use super::registry::BRepRegistry;

pub fn build_edge_curves(_graph: &mut SceneGraph, _reg: &BRepRegistry, _solids: &[SolidKey]) -> usize {
    0 // TODO: implement after build_brep
}
