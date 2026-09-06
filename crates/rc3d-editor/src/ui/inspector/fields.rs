//! Node property fields: two-column aligned grid fed by field descriptors.

use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

use crate::commands::EditorCommand;
use crate::ui::hierarchy;
use slotmap::Key;

pub(super) fn inspector_fields(
    ui: &mut egui::Ui,
    id: NodeId,
    data: &NodeData,
    graph: &SceneGraph,
    push: &mut impl FnMut(EditorCommand),
) {
    ui.label(data.type_name());
    let descriptors = data.field_descriptors();
    if descriptors.is_empty() {
        ui.label(hierarchy::node_type_tag(data));
    } else {
        // Two-column grid: fixed label column keeps every row aligned.
        egui::Grid::new(("inspector_fields", id.data().as_ffi()))
            .num_columns(2)
            .spacing([8.0_f32, 4.0_f32])
            .min_col_width(64.0_f32)
            .max_col_width(110.0_f32)
            .show(ui, |ui| {
                for desc in descriptors {
                    field_row(ui, graph, id, desc.name, desc.field_index, push);
                    ui.end_row();
                }
            });
    }
}

fn field_row(
    ui: &mut egui::Ui,
    graph: &SceneGraph,
    id: NodeId,
    name: &str,
    field_index: u16,
    push: &mut impl FnMut(EditorCommand),
) {
    let Some(value) = rc3d_scene::read_node_field(graph, id, field_index) else {
        ui.label(format!("{name}: (unbound)"));
        return;
    };
    match value {
        rc3d_fields::FieldValue::Bool(mut b) => {
            if ui.checkbox(&mut b, name).changed() {
                push(EditorCommand::SetNodeField {
                    node: id,
                    field_index,
                    value: rc3d_fields::FieldValue::Bool(b),
                });
            }
        }
        rc3d_fields::FieldValue::Int32(mut i) => {
            ui.horizontal(|ui| {
                ui.label(name);
                if ui.add(egui::DragValue::new(&mut i)).changed() {
                    push(EditorCommand::SetNodeField {
                        node: id,
                        field_index,
                        value: rc3d_fields::FieldValue::Int32(i),
                    });
                }
            });
        }
        rc3d_fields::FieldValue::Float(mut f) => {
            ui.horizontal(|ui| {
                ui.label(name);
                if ui.add(egui::DragValue::new(&mut f).speed(0.02)).changed() {
                    push(EditorCommand::SetNodeField {
                        node: id,
                        field_index,
                        value: rc3d_fields::FieldValue::Float(f),
                    });
                }
            });
        }
        rc3d_fields::FieldValue::Vec3f(v) => {
            let mut arr = v.to_array();
            ui.horizontal(|ui| {
                ui.label(name);
                let mut changed = false;
                changed |= ui
                    .add(egui::DragValue::new(&mut arr[0]).speed(0.02))
                    .changed();
                changed |= ui
                    .add(egui::DragValue::new(&mut arr[1]).speed(0.02))
                    .changed();
                changed |= ui
                    .add(egui::DragValue::new(&mut arr[2]).speed(0.02))
                    .changed();
                if name.to_ascii_lowercase().contains("color") {
                    changed |= ui.color_edit_button_rgb(&mut arr).changed();
                }
                if changed {
                    push(EditorCommand::SetNodeField {
                        node: id,
                        field_index,
                        value: rc3d_fields::FieldValue::Vec3f(rc3d_core::math::Vec3::from_array(
                            arr,
                        )),
                    });
                }
            });
        }
        rc3d_fields::FieldValue::Vec2f(v) => {
            let mut arr = v.to_array();
            ui.horizontal(|ui| {
                ui.label(name);
                let mut changed = false;
                changed |= ui
                    .add(egui::DragValue::new(&mut arr[0]).speed(0.02))
                    .changed();
                changed |= ui
                    .add(egui::DragValue::new(&mut arr[1]).speed(0.02))
                    .changed();
                if changed {
                    push(EditorCommand::SetNodeField {
                        node: id,
                        field_index,
                        value: rc3d_fields::FieldValue::Vec2f(rc3d_core::math::Vec2::from_array(
                            arr,
                        )),
                    });
                }
            });
        }
        rc3d_fields::FieldValue::Vec4f(v) => {
            let mut arr = v.to_array();
            ui.horizontal(|ui| {
                ui.label(name);
                let mut changed = false;
                for c in &mut arr {
                    changed |= ui.add(egui::DragValue::new(c).speed(0.02)).changed();
                }
                if changed {
                    push(EditorCommand::SetNodeField {
                        node: id,
                        field_index,
                        value: rc3d_fields::FieldValue::Vec4f(rc3d_core::math::Vec4::from_array(
                            arr,
                        )),
                    });
                }
            });
        }
        rc3d_fields::FieldValue::String(mut s) => {
            ui.horizontal(|ui| {
                ui.label(name);
                if ui.text_edit_singleline(&mut s).changed() {
                    push(EditorCommand::SetNodeField {
                        node: id,
                        field_index,
                        value: rc3d_fields::FieldValue::String(s),
                    });
                }
            });
        }
        other => {
            ui.label(format!("{name}: {other:?}"));
        }
    }
}
