//! Scene hierarchy panel using `egui_ltreeview`.

use std::collections::HashSet;

use egui_ltreeview::{
    Action, DirPosition, IndentHintStyle, NodeBuilder, RowLayout, TreeView, TreeViewBuilder,
    TreeViewState,
};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

use crate::commands::EditorCommand;
use crate::ui::i18n::t;
use crate::ui::icons::Icon;
use crate::ui::theme::ThemePalette;
use crate::ui::types::EditorUiContext;

pub(super) fn draw_hierarchy(
    ui: &mut egui::Ui,
    graph: &SceneGraph,
    ui_ctx: &EditorUiContext,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    let pal = ui_ctx.ui_theme.palette();
    ui.horizontal(|ui| {
        ui.heading(t(loc, "panel.hierarchy"));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let sel = ui_ctx.selected.iter().copied().next();
            if super::icons::icon_button(
                ui,
                Icon::Delete,
                t(loc, "hier.delete"),
                false,
                20.0_f32,
                &pal,
            )
            .clicked()
            {
                for &id in &ui_ctx.selected {
                    push(EditorCommand::DeleteNode(id));
                }
            }
            if super::icons::icon_button(
                ui,
                Icon::Duplicate,
                t(loc, "hier.duplicate"),
                false,
                20.0_f32,
                &pal,
            )
            .clicked()
            {
                for &id in &ui_ctx.selected {
                    push(EditorCommand::DuplicateNode(id));
                }
            }
            let add = super::icons::icon_button(
                ui,
                Icon::Add,
                t(loc, "hier.add_child"),
                false,
                20.0_f32,
                &pal,
            );
            let add_id = ui.make_persistent_id("rc3d_hier_add_root");
            egui::Popup::from_toggle_button_response(&add)
                .id(add_id)
                .close_behavior(egui::PopupCloseBehavior::CloseOnClick)
                .show(|ui| {
                    ui.set_min_width(160.0_f32);
                    super::menus::create_node_menu(ui, sel, loc, push);
                });
        });
    });
    ui.separator();

    let tree_id = ui.make_persistent_id("rc3d_hierarchy");
    let mut state = TreeViewState::<NodeId>::load(ui, tree_id).unwrap_or_default();
    let scene_selected: Vec<NodeId> = ui_ctx.selected.iter().copied().collect();
    let tree_selected: HashSet<NodeId> = state.selected().iter().copied().collect();
    if tree_selected != ui_ctx.selected {
        state.set_selected(scene_selected.clone());
        for id in &scene_selected {
            state.expand_parents_of(id);
        }
    }

    let (_resp, actions) = TreeView::new(tree_id)
        .allow_multi_selection(true)
        .indent_hint_style(IndentHintStyle::Hook)
        .row_layout(RowLayout::AlignedIcons)
        .override_indent(Some(16.0_f32))
        .fallback_context_menu(|ui, nodes| {
            ui.set_min_width(180.0_f32);
            if nodes.is_empty() {
                ui.menu_button(t(loc, "menu.add"), |ui| {
                    super::menus::create_node_menu(ui, None, loc, push);
                });
                return;
            }
            if let Some(&id) = nodes.first() {
                ui.menu_button(t(loc, "hier.add_child"), |ui| {
                    super::menus::create_node_menu(ui, Some(id), loc, push);
                });
            }
            ui.separator();
            if ui.button(t(loc, "hier.duplicate")).clicked() {
                for &id in nodes {
                    push(EditorCommand::DuplicateNode(id));
                }
            }
            if ui.button(t(loc, "hier.delete")).clicked() {
                for &id in nodes {
                    push(EditorCommand::DeleteNode(id));
                }
            }
        })
        .show_state(ui, &mut state, |builder| {
            for &root in graph.roots() {
                add_node(builder, graph, root, pal);
            }
        });
    state.store(ui, tree_id);

    for action in actions {
        match action {
            Action::SetSelected(ids) => {
                push(EditorCommand::SetSelectionMany(ids));
            }
            Action::Drag(dnd) => {
                if !drop_allowed(graph, &dnd.source, dnd.target) {
                    dnd.remove_drop_marker(ui);
                }
            }
            Action::Move(dnd) => {
                if !drop_allowed(graph, &dnd.source, dnd.target) {
                    dnd.remove_drop_marker(ui);
                    continue;
                }
                let Some(index) = drop_index(graph, dnd.target, dnd.position) else {
                    continue;
                };
                push(EditorCommand::ReparentNodes {
                    ids: dnd.source,
                    parent: Some(dnd.target),
                    index,
                });
            }
            Action::Activate(_) | Action::DragExternal(_) | Action::MoveExternal(_) => {}
        }
    }
}

fn drop_allowed(graph: &SceneGraph, sources: &[NodeId], target: NodeId) -> bool {
    for &src in sources {
        if src == target {
            return false;
        }
        let mut walk = Some(target);
        while let Some(n) = walk {
            if n == src {
                return false;
            }
            walk = graph.parent(n);
        }
    }
    true
}

fn drop_index(graph: &SceneGraph, parent: NodeId, position: DirPosition<NodeId>) -> Option<usize> {
    let children = graph.children(parent)?;
    Some(match position {
        DirPosition::First => 0,
        DirPosition::Last => children.len(),
        DirPosition::After(id) => children
            .iter()
            .position(|&c| c == id)
            .map(|i| i + 1)
            .unwrap_or(children.len()),
        DirPosition::Before(id) => children.iter().position(|&c| c == id).unwrap_or(0),
    })
}

fn add_node(
    builder: &mut TreeViewBuilder<'_, NodeId>,
    graph: &SceneGraph,
    id: NodeId,
    pal: ThemePalette,
) {
    let Some(entry) = graph.get(id) else {
        return;
    };
    let label = entry
        .name
        .clone()
        .unwrap_or_else(|| node_type_tag(&entry.data).to_string());
    let icon = type_icon(&entry.data);
    let children: Vec<NodeId> = graph.children(id).unwrap_or(&[]).to_vec();
    builder.node(
        NodeBuilder::dir(id)
            .default_open(true)
            .drop_allowed(true)
            .label(label)
            .icon(move |ui| super::icons::paint_tree_icon(ui, icon, &pal))
            .closer(move |ui, closer| super::icons::paint_tree_closer(ui, closer.is_open, &pal)),
    );
    for c in children {
        add_node(builder, graph, c, pal);
    }
    builder.close_dir();
}

fn type_icon(data: &NodeData) -> Icon {
    match data {
        NodeData::Separator(_)
        | NodeData::Group(_)
        | NodeData::Switch(_)
        | NodeData::Lod(_)
        | NodeData::Environment(_)
        | NodeData::MultipleCopy(_) => Icon::Folder,
        NodeData::DirectionalLight(_)
        | NodeData::PointLight(_)
        | NodeData::SpotLight(_)
        | NodeData::AreaLight(_)
        | NodeData::HemisphereLight(_)
        | NodeData::LightProbe(_) => Icon::Light,
        NodeData::PerspectiveCamera(_)
        | NodeData::OrthographicCamera(_)
        | NodeData::StereoCamera(_)
        | NodeData::CubeCamera(_) => Icon::Camera,
        _ => Icon::Shape,
    }
}

pub(super) fn node_type_tag(data: &NodeData) -> &'static str {
    match data {
        NodeData::Separator(_) => "Separator",
        NodeData::Group(_)
        | NodeData::Environment(_)
        | NodeData::ShapeHints(_)
        | NodeData::Annotation(_)
        | NodeData::AnnotationSet(_)
        | NodeData::ResetTransform(_)
        | NodeData::Texture2Transform(_)
        | NodeData::MaterialBinding(_)
        | NodeData::IndexedLineSet(_)
        | NodeData::File(_)
        | NodeData::Decal(_)
        | NodeData::ExplodedView(_)
        | NodeData::ReflectionPlane(_) => "Group",
        NodeData::CubeCamera(_) => "CubeCamera",
        NodeData::Billboard(_) => "Billboard",
        NodeData::Sprite(_) => "Sprite",
        NodeData::Transform(_) => "Transform",
        NodeData::Rotation(_) => "Rotation",
        NodeData::RotationXYZ(_) => "RotXYZ",
        NodeData::Material(_) => "Material",
        NodeData::Triangle(_) => "Triangle",
        NodeData::Cube(_) => "Cube",
        NodeData::Sphere(_) => "Sphere",
        NodeData::Cone(_) => "Cone",
        NodeData::Cylinder(_) => "Cylinder",
        NodeData::Torus(_) => "Torus",
        NodeData::IndexedFaceSet(_) => "IFS",
        NodeData::SkinnedMesh(_) => "SkinnedMesh",
        NodeData::MorphTarget(_) => "Morph",
        NodeData::RayTracing(_) => "RayTracing",
        NodeData::Volume(_) => "Volume",
        NodeData::PointCloud(_) => "PtCloud",
        NodeData::PerspectiveCamera(_) => "PerspCam",
        NodeData::StereoCamera(_) => "StereoCam",
        NodeData::OrthographicCamera(_) => "OrthoCam",
        NodeData::DirectionalLight(_) => "DirLight",
        NodeData::PointLight(_) => "PointLight",
        NodeData::SpotLight(_) | NodeData::AreaLight(_) => "SpotLight",
        NodeData::HemisphereLight(_) => "HemiLight",
        NodeData::LightProbe(_) => "LightProbe",
        NodeData::HandlerNode(_) => "Handler",
        NodeData::EventCallback(_) => "EventCb",
        NodeData::PickStyle(_) => "PickStyle",
        NodeData::Lod(_) => "LOD",
        NodeData::Switch(_) => "Switch",
        NodeData::MultipleCopy(_) => "MultiCopy",
        NodeData::SectionPlane(_) => "Section",
        NodeData::Text2(_) => "Text2",
        NodeData::Text3(_) => "Text3",
        NodeData::Font(_) => "Font",
        NodeData::Measurement(_) => "Measure",
        NodeData::Markup(_) => "Markup",
        NodeData::Coordinate3(_) => "Coord3",
        NodeData::TextureCoordinate2(_) => "TexCoord2",
        NodeData::Normal(_) => "Normal",
        NodeData::InstancedMesh(_) => "InstancedMesh",
        NodeData::BatchedMesh(_) => "BatchedMesh",
        NodeData::TransformManip(_) => "TransformManip",
        NodeData::Dragger(_) => "Dragger",
        NodeData::Custom(_, d) => d.type_name(),
    }
}
