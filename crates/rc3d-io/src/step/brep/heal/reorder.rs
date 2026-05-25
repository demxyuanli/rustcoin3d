//! Heal: wire edge reordering. T3.1

use super::super::topo::{EdgeKey, Orientation, BRepWire};
use super::super::registry::BRepRegistry;

pub fn reorder_wire_edges(_edges: &[(EdgeKey, Orientation)], _reg: &BRepRegistry) -> Option<Vec<(EdgeKey, Orientation)>> {
    None // TODO
}
