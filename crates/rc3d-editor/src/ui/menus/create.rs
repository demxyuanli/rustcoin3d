//! Create menu: categorized node creation.

use rc3d_core::NodeId;

use crate::commands::EditorCommand;
use crate::ui::i18n::{t, UiLocale};
use crate::ui::types::NodeDataType;

pub(in crate::ui) fn create_node_menu<F: FnMut(EditorCommand)>(
    ui: &mut egui::Ui,
    parent: Option<NodeId>,
    loc: UiLocale,
    push: &mut F,
) {
    let add = |ui: &mut egui::Ui, key: &'static str, ty: NodeDataType, push: &mut F| {
        if ui.button(t(loc, key)).clicked() {
            push(EditorCommand::CreateNode {
                node_type: ty,
                parent,
            });
        }
    };
    ui.menu_button(t(loc, "create.group.shapes"), |ui| {
        add(ui, "create.cube", NodeDataType::Cube, push);
        add(ui, "create.sphere", NodeDataType::Sphere, push);
        add(ui, "create.cylinder", NodeDataType::Cylinder, push);
        add(ui, "create.cone", NodeDataType::Cone, push);
    });
    ui.menu_button(t(loc, "create.group.grouping"), |ui| {
        add(ui, "create.separator", NodeDataType::Separator, push);
        add(ui, "create.transform", NodeDataType::Transform, push);
        add(ui, "create.material", NodeDataType::Material, push);
        add(ui, "create.switch", NodeDataType::Switch, push);
        add(ui, "create.lod", NodeDataType::Lod, push);
        add(ui, "create.environment", NodeDataType::Environment, push);
    });
    ui.menu_button(t(loc, "create.group.lights"), |ui| {
        add(ui, "create.dir_light", NodeDataType::DirectionalLight, push);
        add(ui, "create.point_light", NodeDataType::PointLight, push);
        add(ui, "create.spot_light", NodeDataType::SpotLight, push);
        add(ui, "create.hemi_light", NodeDataType::HemisphereLight, push);
        add(ui, "create.area_light", NodeDataType::AreaLight, push);
        add(ui, "create.light_probe", NodeDataType::LightProbe, push);
    });
    ui.menu_button(t(loc, "create.group.cameras"), |ui| {
        add(
            ui,
            "create.persp_cam",
            NodeDataType::PerspectiveCamera,
            push,
        );
        add(
            ui,
            "create.ortho_cam",
            NodeDataType::OrthographicCamera,
            push,
        );
        add(ui, "create.stereo_cam", NodeDataType::StereoCamera, push);
    });
    ui.menu_button(t(loc, "create.group.text"), |ui| {
        add(ui, "create.text2", NodeDataType::Text2, push);
        add(ui, "create.text3", NodeDataType::Text3, push);
        add(ui, "create.font", NodeDataType::Font, push);
        add(ui, "create.billboard", NodeDataType::Billboard, push);
        add(ui, "create.sprite", NodeDataType::Sprite, push);
    });
    ui.menu_button(t(loc, "create.group.fx"), |ui| {
        add(ui, "create.section", NodeDataType::SectionPlane, push);
        add(ui, "create.annotation", NodeDataType::AnnotationSet, push);
        add(ui, "create.measurement", NodeDataType::Measurement, push);
        add(ui, "create.markup", NodeDataType::Markup, push);
        add(ui, "create.particles", NodeDataType::Particles, push);
        add(ui, "create.instanced", NodeDataType::InstancedMesh, push);
        add(ui, "create.batched", NodeDataType::BatchedMesh, push);
    });
}
