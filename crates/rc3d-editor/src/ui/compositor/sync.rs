//! Graph <-> egui-snarl conversion. UI widgets stay in `mod.rs`.

use std::collections::HashMap;

use egui::pos2;
use egui_snarl::{InPinId, OutPinId, Snarl};

use rc3d_render::{CompEdge, CompNode, CompNodeId, CompOp, CompositorGraph};

pub(super) fn protected(op: CompOp) -> bool {
    matches!(op, CompOp::RenderLayers | CompOp::Viewer)
}

pub(super) fn snarl_from_graph(graph: &CompositorGraph) -> Snarl<CompNode> {
    let mut snarl = Snarl::new();
    let mut map = HashMap::new();
    for n in &graph.nodes {
        // Persisted fold state drives snarl's open flag; snarl defaults new
        // nodes to open so this keeps collapsed nodes collapsed on reload.
        let id = snarl.insert_node(pos2(n.pos[0], n.pos[1]), n.clone());
        snarl.open_node(id, n.open);
        map.insert(n.id, id);
    }
    for e in &graph.edges {
        let Some(&from) = map.get(&e.from) else {
            continue;
        };
        let Some(&to) = map.get(&e.to) else {
            continue;
        };
        let Some(dst) = graph.node(e.to) else {
            continue;
        };
        let _ = snarl.connect(
            OutPinId {
                node: from,
                output: 0,
            },
            InPinId {
                node: to,
                input: snarl_slot(dst.op, e.to_slot),
            },
        );
    }
    snarl
}

pub(super) fn export_graph(snarl: &Snarl<CompNode>) -> (Vec<CompNode>, Vec<CompEdge>) {
    let mut nodes = Vec::new();
    for (id, pos, n) in snarl.nodes_pos_ids() {
        let mut node = n.clone();
        node.id = id.0 as CompNodeId;
        node.pos = [pos.x, pos.y];
        // Fold state lives on snarl's Node wrapper, not the CompNode value.
        node.open = snarl.get_node_info(id).is_some_and(|info| info.open);
        nodes.push(node);
    }
    let mut edges = Vec::new();
    for (from, to) in snarl.wires() {
        let Some(dst) = snarl.get_node(to.node) else {
            continue;
        };
        let Some(slot) = graph_slot(dst.op, to.input) else {
            continue;
        };
        edges.push(CompEdge {
            from: from.node.0 as CompNodeId,
            to: to.node.0 as CompNodeId,
            to_slot: slot,
        });
    }
    (nodes, edges)
}

/// UI input sockets are exactly the *connectable* graph inputs, in order.
/// Every displayed input pin has a graph slot, so snarl pin index == slot.
pub(super) fn graph_slot(op: CompOp, snarl_input: usize) -> Option<u8> {
    (snarl_input < ui_input_count(op)).then_some(snarl_input as u8)
}

fn snarl_slot(_op: CompOp, graph_slot: u8) -> usize {
    graph_slot as usize
}

pub(super) fn ui_input_count(op: CompOp) -> usize {
    match op {
        CompOp::RenderLayers | CompOp::Rgb | CompOp::Value => 0,
        CompOp::Mix | CompOp::AlphaOver | CompOp::Math => 2,
        _ => 1,
    }
}

/// First image input socket (all connectable inputs start at 0).
pub(super) fn first_image_pin(_op: CompOp) -> usize {
    0
}
