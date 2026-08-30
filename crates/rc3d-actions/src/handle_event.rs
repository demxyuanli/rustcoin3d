//! HandleEventAction — traverses the scene graph routing events to nodes.
//!
//! Follows the Coin3D SoHandleEventAction pattern: pick-first traversal for pointer
//! events; full-tree discovery of `EventCallback` nodes for keyboard/scroll/touch.

use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

use crate::action::{Action, ActionKind};
use crate::event::{Event, EventContext};
use crate::ray_pick::{PickHit, RayPickAction};

/// Traverses the scene graph and routes events to nodes.
pub struct HandleEventAction {
    pub ctx: EventContext,
    pub hit_node: Option<NodeId>,
    pub hit_point: Option<rc3d_core::math::Vec3>,
    pub pick_hits: Vec<PickHit>,
    /// `EventCallback` node ids (for non-pointer events or custom dispatch).
    pub event_callback_nodes: Vec<NodeId>,
}

impl HandleEventAction {
    pub fn new(ctx: EventContext) -> Self {
        Self {
            ctx,
            hit_node: None,
            hit_point: None,
            pick_hits: Vec::new(),
            event_callback_nodes: Vec::new(),
        }
    }

    fn pick_first(&mut self, graph: &SceneGraph, root: NodeId, screen_x: f32, screen_y: f32, vp_w: f32, vp_h: f32) {
        let ray = self.ctx.pick_ray(screen_x, screen_y, vp_w, vp_h);
        let mut picker = RayPickAction::new(ray);
        picker.apply(graph, root);
        self.pick_hits = picker.hits;

        if let Some(hit) = self.pick_hits.first() {
            self.hit_node = Some(hit.node);
            self.hit_point = Some(hit.point);
        }
    }

    fn collect_event_callback_nodes(&mut self, graph: &SceneGraph, node: NodeId) {
        let Some(entry) = graph.get(node) else {
            return;
        };
        self.record_event_callback(node, &entry.data);
        for &c in &entry.children {
            self.collect_event_callback_nodes(graph, c);
        }
    }

    fn record_event_callback(&mut self, node: NodeId, data: &NodeData) {
        let NodeData::EventCallback(ec) = data else {
            return;
        };
        if !ec.enabled {
            return;
        }
        self.event_callback_nodes.push(node);
        if ec.consume {
            self.ctx.consume();
        }
    }

    /// Coin3D pick-path: the hit node, its ancestors, and EventCallback siblings
    /// of nodes on that path (same Separator as the picked shape).
    fn collect_callbacks_on_pick_path(&mut self, graph: &SceneGraph, hit: NodeId) {
        self.record_event_callback_at(graph, hit);
        let mut child = hit;
        let mut cur = graph.parent(hit);
        while let Some(id) = cur {
            self.record_event_callback_at(graph, id);
            if let Some(entry) = graph.get(id) {
                for &sib in &entry.children {
                    if sib != child {
                        self.record_event_callback_at(graph, sib);
                    }
                }
            }
            child = id;
            cur = graph.parent(id);
        }
    }

    fn record_event_callback_at(&mut self, graph: &SceneGraph, node: NodeId) {
        if let Some(entry) = graph.get(node) {
            self.record_event_callback(node, &entry.data);
        }
    }
}

impl Action for HandleEventAction {
    fn kind(&self) -> ActionKind {
        ActionKind::HandleEvent
    }

    fn apply(&mut self, graph: &SceneGraph, root: NodeId) {
        self.event_callback_nodes.clear();
        self.hit_node = None;
        self.hit_point = None;
        self.pick_hits.clear();

        match &self.ctx.event {
            Event::KeyPress { .. } | Event::KeyRelease { .. } | Event::Scroll { .. } | Event::Touch { .. } => {
                self.collect_event_callback_nodes(graph, root);
            }
            Event::MouseMove { x, y, .. } => {
                let (pw, ph) = self.ctx.pointer_pick_viewport.unwrap_or((1.0, 1.0));
                self.pick_first(graph, root, *x, *y, pw, ph);
                if let Some(hit) = self.hit_node {
                    self.collect_callbacks_on_pick_path(graph, hit);
                }
            }
            Event::ButtonPress { x, y, .. } | Event::ButtonRelease { x, y, .. } => {
                let (pw, ph) = self.ctx.pointer_pick_viewport.unwrap_or((1.0, 1.0));
                self.pick_first(graph, root, *x, *y, pw, ph);
                if let Some(hit) = self.hit_node {
                    self.collect_callbacks_on_pick_path(graph, hit);
                }
            }
        }
    }
}

/// Optional API: only collect `EventCallback` nodes without picking (same as non-pointer path).
impl HandleEventAction {
    pub fn apply_non_pointer_only(&mut self, graph: &SceneGraph, root: NodeId) {
        self.event_callback_nodes.clear();
        self.collect_event_callback_nodes(graph, root);
    }
}
