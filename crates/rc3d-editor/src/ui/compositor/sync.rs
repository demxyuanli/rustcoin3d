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

pub(super) fn graph_slot(op: CompOp, snarl_input: usize) -> Option<u8> {
    match op {
        CompOp::Mix | CompOp::AlphaOver => match snarl_input {
            1 => Some(0),
            2 => Some(1),
            _ => None,
        },
        // Math: A and B are both wire-able, each with a scalar fallback value.
        CompOp::Math => match snarl_input {
            0 => Some(0),
            1 => Some(1),
            _ => None,
        },
        CompOp::RenderLayers | CompOp::Rgb | CompOp::Value => None,
        _ => (snarl_input == 0).then_some(0),
    }
}

fn snarl_slot(op: CompOp, graph_slot: u8) -> usize {
    match op {
        CompOp::Mix | CompOp::AlphaOver => graph_slot as usize + 1,
        _ => graph_slot as usize,
    }
}

pub(super) fn ui_input_count(op: CompOp) -> usize {
    match op {
        CompOp::RenderLayers | CompOp::Rgb | CompOp::Value => 0,
        CompOp::Mix | CompOp::AlphaOver | CompOp::BrightContrast => 3,
        CompOp::Blur | CompOp::Bloom | CompOp::Math => 2,
        _ => 1,
    }
}

pub(super) fn first_image_pin(op: CompOp) -> usize {
    match op {
        CompOp::Mix | CompOp::AlphaOver => 1,
        _ => 0,
    }
}
