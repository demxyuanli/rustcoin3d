//! Scene hierarchy panel using `egui_ltreeview`.

use egui_ltreeview::{
    Action, DirPosition, IndentHintStyle, NodeBuilder, RowLayout, TreeView, TreeViewBuilder,
    TreeViewState,
};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use crate::commands::EditorCommand;
use crate::ui::i18n::t;
use crate::ui::icons::Icon;
use crate::ui::theme::ThemePalette;
use crate::ui::types::EditorUiContext;

pub(super) fn draw_hierarchy(
    ui: &mut egui::Ui,
    graph: &SceneGraph,
    ui_ctx: &EditorUiContext,
    chrome: &mut crate::ui::types::EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    let pal = ui_ctx.ui_theme.palette();
    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(&mut chrome.outliner_filter)
                .hint_text(t(loc, "hier.filter"))
                .desired_width(140.0_f32),
        );
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

    let (_resp, actions) = {
        let queued = Rc::new(RefCell::new(Vec::<EditorCommand>::new()));
        let (_tree_resp, actions) = egui::ScrollArea::both()
            .id_salt("hierarchy_tree_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                TreeView::new(tree_id)
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
                        super::menus::hierarchy_node_context_actions(ui, nodes, loc, &pal, push);
                        ui.separator();
                        let _ = super::menus::selection_context_actions(ui, loc, &pal, push);
                    })
                    .show_state(ui, &mut state, |builder| {
                        let filter = chrome.outliner_filter.to_ascii_lowercase();
                        for &root in graph.roots() {
                            add_node(
                                builder,
                                graph,
                                root,
                                pal,
                                ui_ctx,
                                &filter,
                                loc,
                                queued.clone(),
                            );
                        }
                    })
            })
            .inner;
        for cmd in queued.borrow_mut().drain(..) {
            push(cmd);
        }
        (_tree_resp, actions)
    };
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
    ui_ctx: &EditorUiContext,
    filter: &str,
    loc: crate::ui::i18n::UiLocale,
    queued: Rc<RefCell<Vec<EditorCommand>>>,
) {
    let Some(entry) = graph.get(id) else {
        return;
    };
    let label = entry
        .name
        .clone()
        .unwrap_or_else(|| node_type_tag(&entry.data).to_string());
    if !filter.is_empty() && !node_matches(graph, id, filter) {
        return;
    }
    let hidden = ui_ctx.hidden_nodes.contains(&id);
    let locked = ui_ctx.locked_nodes.contains(&id);
    let icon = type_icon(&entry.data);
    let children: Vec<NodeId> = graph.children(id).unwrap_or(&[]).to_vec();
    let q = queued.clone();
    builder.node(
        NodeBuilder::dir(id)
            .default_open(true)
            .drop_allowed(true)
            .icon(move |ui| super::icons::paint_tree_icon(ui, icon, &pal))
            .closer(move |ui, closer| super::icons::paint_tree_closer(ui, closer.is_open, &pal))
            .label_ui(move |ui| {
                row_label(ui, id, &label, hidden, locked, loc, &pal, q.clone());
            }),
    );
    for c in children {
        add_node(builder, graph, c, pal, ui_ctx, filter, loc, queued.clone());
    }
    builder.close_dir();
}

const ROW_BTN: f32 = 16.0;

fn row_label(
    ui: &mut egui::Ui,
    id: NodeId,
    label: &str,
    hidden: bool,
    locked: bool,
    loc: crate::ui::i18n::UiLocale,
    pal: &ThemePalette,
    queued: Rc<RefCell<Vec<EditorCommand>>>,
) {
    // Blender-outliner style: buttons right-aligned (RTL), revealed only
    // while the pointer is over the row. Hit-testing uses the raw pointer
    // position (same pattern as egui_ltreeview's own closer hover) because
    // `rect_contains_pointer` relies on an Area layer hit-test that child
    // Uis inside the tree never register.
    ui.add(egui::Label::new(label).selectable(false));
    // The label scope's max_rect spans from the label to the row's right
    // edge at the row's height — the row hover region.
    let hover_rect = ui.max_rect();
    let hovered = ui
        .input(|i| i.pointer.latest_pos())
        .is_some_and(|pos| hover_rect.contains(pos));
    if hovered {
        // Detached child Ui: does not advance the parent cursor nor feed the
        // tree's content-width measurement, so revealed buttons can never
        // widen rows or shift the hover region.
        let mut btn_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(hover_rect)
                .layout(egui::Layout::right_to_left(egui::Align::Center)),
        );
        btn_ui.spacing_mut().item_spacing.x = 2.0_f32;
        if super::icons::icon_button(
            &mut btn_ui,
            Icon::Cursor,
            t(loc, "tools.select"),
            false,
            ROW_BTN,
            pal,
        )
        .clicked()
        {
            queued
                .borrow_mut()
                .push(EditorCommand::SetSelection(Some(id)));
        }
        if super::icons::icon_button(
            &mut btn_ui,
            Icon::Lock,
            t(loc, "key.lock"),
            locked,
            ROW_BTN,
            pal,
        )
        .clicked()
        {
            queued.borrow_mut().push(EditorCommand::ToggleLockNode(id));
        }
        let eye = if hidden { Icon::EyeOff } else { Icon::Eye };
        if super::icons::icon_button(
            &mut btn_ui,
            eye,
            t(loc, "insp.visible"),
            hidden,
            ROW_BTN,
            pal,
        )
        .clicked()
        {
            queued
                .borrow_mut()
                .push(EditorCommand::SetNodeVisibility(id, !hidden));
        }
    }
}

fn node_matches(graph: &SceneGraph, id: NodeId, filter: &str) -> bool {
    let Some(entry) = graph.get(id) else {
        return false;
    };
    let label = entry
        .name
        .as_deref()
        .unwrap_or_else(|| node_type_tag(&entry.data));
    if label.to_ascii_lowercase().contains(filter) {
        return true;
    }
    if let Some(children) = graph.children(id) {
        return children.iter().any(|&c| node_matches(graph, c, filter));
    }
    false
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
