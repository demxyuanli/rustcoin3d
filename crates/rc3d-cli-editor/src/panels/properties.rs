use rc3d_core::math::Quat;
use rc3d_scene::node_data::NodeData;
use rc3d_scene::SceneGraph;

use crate::engine::state::EngineState;

pub struct PropertiesPanel;

impl PropertiesPanel {
    pub fn new() -> Self {
        Self
    }

    pub fn ui(&self, ui: &mut egui::Ui, state: &EngineState) {
        egui::Panel::right("properties")
            .min_size(220.0)
            .resizable(true)
            .show(ui, |ui| {
                ui.heading("Properties");
                ui.separator();

                if state.selection.is_empty() {
                    self.show_scene_summary(ui, state);
                } else {
                    self.show_selection(ui, state);
                }
            });
    }

    fn show_scene_summary(&self, ui: &mut egui::Ui, state: &EngineState) {
        ui.label("Scene Summary");
        ui.separator();
        let roots = state.scene.roots();
        ui.label(format!("Root nodes: {}", roots.len()));
        ui.label(format!("Total nodes: {}", count_all_nodes(&state.scene)));
        ui.label(format!("Selection: {} node(s)", state.selection.len()));
        ui.label(format!("Display mode: {}", state.display_mode));
        ui.label(format!(
            "Camera: dist={:.1} phi={:.2} theta={:.2}",
            state.camera.distance, state.camera.phi, state.camera.theta
        ));
    }

    fn show_selection(&self, ui: &mut egui::Ui, state: &EngineState) {
        for &node_id in &state.selection {
            ui.collapsing(format!("Node {:?}", node_id), |ui| {
                if let Some(entry) = state.scene.get(node_id) {
                    ui.label(format!("Type: {}", entry.data.type_name()));
                    if let Some(ref name) = entry.name {
                        ui.label(format!("Name: {name}"));
                    }
                    self.show_node_fields(ui, node_id, &entry.data);
                } else {
                    ui.label("(not found)");
                }
            });
        }
    }

    fn show_node_fields(&self, ui: &mut egui::Ui, _node_id: rc3d_core::NodeId, data: &NodeData) {
        match data {
            NodeData::Transform(t) => {
                ui.label("Translation:");
                ui.label(format!(
                    "  [{:.2}, {:.2}, {:.2}]",
                    t.translation.x, t.translation.y, t.translation.z
                ));
                let q = Quat::from_mat4(&t.rotation);
                let (axis, angle) = q.to_axis_angle();
                ui.label(format!(
                    "Rotation: [{:.2}, {:.2}, {:.2}] @ {:.2}",
                    axis.x, axis.y, axis.z, angle
                ));
                ui.label(format!(
                    "Scale: [{:.2}, {:.2}, {:.2}]",
                    t.scale.x, t.scale.y, t.scale.z
                ));
                ui.label("Use: prop set <id> tx/ty/tz/sx/sy/sz <value>");
            }
            NodeData::Material(m) => {
                ui.label(format!("Base: [{:.2}, {:.2}, {:.2}]", m.base_color.x, m.base_color.y, m.base_color.z));
                ui.label(format!("Metallic: {:.2}", m.metallic));
                ui.label(format!("Roughness: {:.2}", m.roughness));
                ui.label(format!("Opacity: {:.2}", m.opacity));
                ui.label(format!("Anisotropic: {:.2}", m.anisotropic));
                ui.label(format!("Alpha: {:?}", m.alpha_mode));
                ui.label("Use: prop set <id> roughness/metallic/opacity <value>");
            }
            NodeData::Cube(c) => {
                ui.label(format!("Size: [{:.2}, {:.2}, {:.2}]", c.width, c.height, c.depth));
            }
            NodeData::Sphere(s) => {
                ui.label(format!("Radius: {:.2}", s.radius));
                ui.label("Use: prop set <id> radius <value>");
            }
            NodeData::DirectionalLight(l) => {
                ui.label(format!("Dir: [{:.2}, {:.2}, {:.2}]", l.direction.x, l.direction.y, l.direction.z));
                ui.label(format!("Color: [{:.2}, {:.2}, {:.2}]", l.color.x, l.color.y, l.color.z));
                ui.label(format!("Intensity: {:.2}", l.intensity));
            }
            NodeData::PerspectiveCamera(c) => {
                ui.label(format!("FOV: {:.2}", c.fov));
                ui.label(format!("Near/Far: {:.2}/{:.2}", c.near, c.far));
            }
            _ => {
                ui.label("(no editable fields)");
            }
        }
    }
}

fn count_all_nodes(graph: &SceneGraph) -> usize {
    let mut count = 0;
    let mut stack: Vec<_> = graph.roots().to_vec();
    while let Some(node) = stack.pop() {
        count += 1;
        if let Some(entry) = graph.get(node) {
            stack.extend(entry.children.iter().copied());
        }
    }
    count
}
