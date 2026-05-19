use rc3d_core::math::Vec2;
use rc3d_core::NodeId;
use rc3d_scene::node_data::{MarkupElement, MarkupNode};
use rc3d_scene::SceneGraph;

/// Markup tool modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkupTool {
    Select,
    Line,
    Rect,
    Circle,
    Freehand,
}

/// Interactive markup drawing state machine.
pub struct MarkupAction {
    pub tool: MarkupTool,
    pub target_node: Option<NodeId>,
    pub click_points: Vec<Vec2>,
    pub preview_element: Option<MarkupElement>,
    pub current_mouse: Vec2,
}

impl Default for MarkupAction {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkupAction {
    pub fn new() -> Self {
        Self {
            tool: MarkupTool::Select,
            target_node: None,
            click_points: Vec::new(),
            preview_element: None,
            current_mouse: Vec2::ZERO,
        }
    }

    pub fn set_tool(&mut self, tool: MarkupTool) {
        self.tool = tool;
        self.cancel();
    }

    /// Ensure there is an active MarkupNode in the scene graph, creating one if needed.
    pub fn ensure_target_node(&mut self, graph: &mut SceneGraph, root: NodeId) {
        if self.target_node.is_none()
            || self.target_node.map_or(true, |id| graph.get(id).is_none())
        {
            let id = graph.insert_child(root, 0, rc3d_scene::NodeData::Markup(MarkupNode::default()));
            self.target_node = Some(id);
        }
    }

    /// Handle mouse down. Returns true if the event was consumed.
    pub fn on_mouse_down(&mut self, screen_pos: Vec2, _graph: &mut SceneGraph) -> bool {
        match self.tool {
            MarkupTool::Line | MarkupTool::Rect | MarkupTool::Circle => {
                self.click_points.push(screen_pos);
                self.update_preview();
                true
            }
            MarkupTool::Freehand => {
                self.click_points.push(screen_pos);
                self.update_preview();
                true
            }
            MarkupTool::Select => false,
        }
    }

    /// Handle mouse move (updates preview).
    pub fn on_mouse_move(&mut self, screen_pos: Vec2) {
        self.current_mouse = screen_pos;
        if !self.click_points.is_empty() {
            self.update_preview();
        }
    }

    /// Handle mouse up. Returns Some(MarkupElement) if an element was completed.
    pub fn on_mouse_up(&mut self, screen_pos: Vec2) -> Option<MarkupElement> {
        self.current_mouse = screen_pos;
        let element = match self.tool {
            MarkupTool::Line => {
                if !self.click_points.is_empty() {
                    let start = self.click_points[0];
                    Some(MarkupElement::Line {
                        start: [start.x, start.y],
                        end: [screen_pos.x, screen_pos.y],
                        color: [1.0, 0.0, 0.0, 0.8],
                        width: 2.0,
                    })
                } else {
                    None
                }
            }
            MarkupTool::Rect => {
                if !self.click_points.is_empty() {
                    let origin = self.click_points[0];
                    let size = [screen_pos.x - origin.x, screen_pos.y - origin.y];
                    Some(MarkupElement::Rect {
                        origin: [origin.x.min(screen_pos.x), origin.y.min(screen_pos.y)],
                        size: [size[0].abs(), size[1].abs()],
                        color: [1.0, 0.0, 0.0, 0.6],
                        filled: false,
                    })
                } else {
                    None
                }
            }
            MarkupTool::Circle => {
                if !self.click_points.is_empty() {
                    let center = self.click_points[0];
                    let radius = (screen_pos - center).length();
                    Some(MarkupElement::Circle {
                        center: [center.x, center.y],
                        radius,
                        color: [1.0, 0.0, 0.0, 0.6],
                    })
                } else {
                    None
                }
            }
            MarkupTool::Freehand => {
                if self.click_points.len() >= 2 {
                    let points: Vec<[f32; 2]> =
                        self.click_points.iter().map(|p| [p.x, p.y]).collect();
                    Some(MarkupElement::Freehand {
                        points,
                        color: [1.0, 0.0, 0.0, 0.8],
                        width: 2.0,
                    })
                } else {
                    None
                }
            }
            MarkupTool::Select => None,
        };

        if element.is_some() {
            self.click_points.clear();
            self.preview_element = None;
        }
        element
    }

    /// Cancel the current operation, clearing temp state.
    pub fn cancel(&mut self) {
        self.click_points.clear();
        self.preview_element = None;
    }

    fn update_preview(&mut self) {
        self.preview_element = match &self.tool {
            MarkupTool::Line if !self.click_points.is_empty() => {
                let start = self.click_points[0];
                Some(MarkupElement::Line {
                    start: [start.x, start.y],
                    end: [self.current_mouse.x, self.current_mouse.y],
                    color: [1.0, 0.0, 0.0, 0.5],
                    width: 1.0,
                })
            }
            MarkupTool::Rect if !self.click_points.is_empty() => {
                let origin = self.click_points[0];
                let size = [
                    self.current_mouse.x - origin.x,
                    self.current_mouse.y - origin.y,
                ];
                Some(MarkupElement::Rect {
                    origin: [
                        origin.x.min(self.current_mouse.x),
                        origin.y.min(self.current_mouse.y),
                    ],
                    size: [size[0].abs(), size[1].abs()],
                    color: [1.0, 0.0, 0.0, 0.4],
                    filled: false,
                })
            }
            MarkupTool::Circle if !self.click_points.is_empty() => {
                let center = self.click_points[0];
                let radius = (self.current_mouse - center).length();
                Some(MarkupElement::Circle {
                    center: [center.x, center.y],
                    radius,
                    color: [1.0, 0.0, 0.0, 0.4],
                })
            }
            _ => None,
        };
    }
}
