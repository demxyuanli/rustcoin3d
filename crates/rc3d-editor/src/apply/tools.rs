use rc3d_core::math::Vec3;
use rc3d_engine_api::Engine;
use rc3d_scene::node_data::{MeasurementType, NodeData};

use crate::commands::EditorCommand;
use crate::context::EditorInteractionState;
use crate::ui::types::CanvasTool;

use super::EditorSession;

pub(super) fn apply(
    engine: &mut Engine,
    interaction: &mut EditorInteractionState,
    session: &mut EditorSession,
    cmd: EditorCommand,
) {
    match cmd {
        EditorCommand::SetGizmoMode(mode) => {
            leave_canvas(engine, interaction, CanvasTool::Transform);
            engine.set_gizmo_mode(mode);
        }
        EditorCommand::SetWalkMode(enabled) => {
            if enabled {
                leave_canvas(engine, interaction, CanvasTool::Walk);
            } else if interaction.canvas == CanvasTool::Walk {
                leave_canvas(engine, interaction, CanvasTool::Select);
            }
            engine.controller.walk_mode = enabled;
            if let Some(vc) = engine.viewport_cameras.active_mut() {
                vc.controller.walk_mode = enabled;
            }
        }
        EditorCommand::ToggleSectionEdit => {
            let on = !interaction.section_edit_mode;
            apply_section_edit(engine, interaction, on);
        }
        EditorCommand::SetSectionEdit(on) => {
            apply_section_edit(engine, interaction, on);
        }
        EditorCommand::SetSelectKind(kind) => {
            leave_canvas(engine, interaction, CanvasTool::Select);
            interaction.select_kind = kind;
        }
        EditorCommand::ToggleMeasurement => {
            if interaction.measurement_mode {
                leave_canvas(engine, interaction, CanvasTool::Select);
            } else {
                leave_canvas(engine, interaction, CanvasTool::Measure);
                interaction.measurement_mode = true;
                let ty = interaction
                    .measurement_type
                    .unwrap_or(MeasurementType::Distance);
                interaction.measurement_type = Some(ty);
                interaction.measure = crate::measurement::action_for(ty);
                interaction.measure_label.clear();
            }
        }
        EditorCommand::SetMeasurementMode(mode) => {
            if let Some(ty) = mode {
                leave_canvas(engine, interaction, CanvasTool::Measure);
                interaction.measurement_mode = true;
                interaction.measurement_type = Some(ty);
                interaction.measure = crate::measurement::action_for(ty);
                interaction.measure_label.clear();
            } else if interaction.canvas == CanvasTool::Measure {
                leave_canvas(engine, interaction, CanvasTool::Select);
            }
        }
        EditorCommand::SaveBookmark(slot) => {
            engine.controller.save_bookmark(slot, bookmark_name(slot));
        }
        EditorCommand::RecallBookmark(slot) => {
            engine.controller.recall_bookmark(slot);
        }
        EditorCommand::SetMarkupTool(tool) => {
            if tool == rc3d_actions::MarkupTool::Select {
                if interaction.canvas == CanvasTool::Markup {
                    leave_canvas(engine, interaction, CanvasTool::Select);
                } else {
                    interaction.markup.set_tool(tool);
                }
            } else {
                leave_canvas(engine, interaction, CanvasTool::Markup);
                interaction.markup.set_tool(tool);
            }
        }
        EditorCommand::MarkupMouseDown { screen_pos } => {
            markup_down(engine, interaction, screen_pos);
        }
        EditorCommand::MarkupMouseMove { screen_pos } => {
            interaction
                .markup
                .on_mouse_move(rc3d_core::math::Vec2::from_array(screen_pos));
            push_markup_preview(engine, interaction);
        }
        EditorCommand::MarkupMouseUp { screen_pos } => {
            markup_up(engine, session, interaction, screen_pos);
        }
        EditorCommand::ClearAllMarkup { node } => {
            pop_markup_preview(engine, interaction);
            if let Some(e) = engine.world.graph.get_mut(node) {
                if let NodeData::Markup(m) = &mut e.data {
                    m.elements.clear();
                    session.dirty = true;
                }
            }
        }
        EditorCommand::MeasurementPick { world } => {
            measurement_pick(engine, session, interaction, world);
        }
        EditorCommand::CancelTool => {
            cancel_tool(engine, interaction);
        }
        EditorCommand::HideSelected => {
            for &id in engine.world.graph.selected_nodes() {
                engine.hidden_nodes.insert(id);
            }
        }
        EditorCommand::IsolateSelected => {
            let keep: std::collections::HashSet<_> = engine
                .world
                .graph
                .selected_nodes()
                .iter()
                .copied()
                .collect();
            engine.hidden_nodes.clear();
            for id in engine.world.graph.all_node_ids() {
                if !keep.contains(&id) {
                    engine.hidden_nodes.insert(id);
                }
            }
        }
        EditorCommand::RevealHidden => {
            engine.hidden_nodes.clear();
        }
        EditorCommand::ToggleLockSelected => {
            let selected: Vec<_> = engine
                .world
                .graph
                .selected_nodes()
                .iter()
                .copied()
                .collect();
            let all_locked = !selected.is_empty()
                && selected
                    .iter()
                    .all(|id| interaction.locked_nodes.contains(id));
            for id in selected {
                if all_locked {
                    interaction.locked_nodes.remove(&id);
                } else {
                    interaction.locked_nodes.insert(id);
                }
            }
        }
        EditorCommand::ToggleLockNode(id) => {
            if interaction.locked_nodes.contains(&id) {
                interaction.locked_nodes.remove(&id);
            } else {
                interaction.locked_nodes.insert(id);
            }
        }
        EditorCommand::HistoryJump { undo_len } => {
            session
                .history
                .jump_to_undo_len(undo_len, &mut engine.world.graph);
            rc3d_engine_api::sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
        }
        EditorCommand::BindKey { action, chord } => {
            session.keymap.set(action, chord);
        }
        _ => {}
    }
}

fn apply_section_edit(engine: &mut Engine, interaction: &mut EditorInteractionState, on: bool) {
    if on {
        leave_canvas(engine, interaction, CanvasTool::Section);
        interaction.section_edit_mode = true;
    } else {
        if interaction.canvas == CanvasTool::Section {
            leave_canvas(engine, interaction, CanvasTool::Select);
        }
        interaction.section_edit_mode = false;
        interaction.section_drag = None;
        interaction.section_hovered = None;
    }
}

fn leave_canvas(engine: &mut Engine, interaction: &mut EditorInteractionState, keep: CanvasTool) {
    if keep != CanvasTool::Select {
        interaction.select_kind = crate::ui::types::SelectKind::Pick;
    }
    if keep != CanvasTool::Measure {
        interaction.measurement_mode = false;
        interaction.measurement_type = None;
        interaction.measure =
            rc3d_actions::MeasurementAction::new(rc3d_actions::MeasurementMode::Distance);
        interaction.measure_label.clear();
    }
    if keep != CanvasTool::Section {
        interaction.section_edit_mode = false;
        interaction.section_drag = None;
        interaction.section_hovered = None;
    }
    if keep != CanvasTool::Markup {
        pop_markup_preview(engine, interaction);
        interaction
            .markup
            .set_tool(rc3d_actions::MarkupTool::Select);
    }
    if keep != CanvasTool::Walk {
        engine.controller.walk_mode = false;
        if let Some(vc) = engine.viewport_cameras.active_mut() {
            vc.controller.walk_mode = false;
        }
    }
    interaction.canvas = keep;
}

fn bookmark_name(slot: usize) -> &'static str {
    match slot {
        0 => "1",
        1 => "2",
        2 => "3",
        3 => "4",
        4 => "5",
        5 => "6",
        6 => "7",
        7 => "8",
        _ => "9",
    }
}

fn markup_down(
    engine: &mut Engine,
    interaction: &mut EditorInteractionState,
    screen_pos: [f32; 2],
) {
    let root = engine.world.graph.roots().first().copied();
    if let Some(root) = root {
        interaction
            .markup
            .ensure_target_node(&mut engine.world.graph, root);
    }
    let _ = interaction.markup.on_mouse_down(
        rc3d_core::math::Vec2::from_array(screen_pos),
        &mut engine.world.graph,
    );
    push_markup_preview(engine, interaction);
}

fn markup_up(
    engine: &mut Engine,
    session: &mut EditorSession,
    interaction: &mut EditorInteractionState,
    screen_pos: [f32; 2],
) {
    pop_markup_preview(engine, interaction);
    if let Some(el) = interaction
        .markup
        .on_mouse_up(rc3d_core::math::Vec2::from_array(screen_pos))
    {
        if let Some(id) = interaction.markup.target_node {
            if let Some(e) = engine.world.graph.get_mut(id) {
                if let NodeData::Markup(m) = &mut e.data {
                    m.elements.push(el);
                    session.dirty = true;
                }
            }
        }
    }
}

fn pop_markup_preview(engine: &mut Engine, interaction: &mut EditorInteractionState) {
    if !interaction.markup_preview_live {
        return;
    }
    if let Some(id) = interaction.markup.target_node {
        if let Some(e) = engine.world.graph.get_mut(id) {
            if let NodeData::Markup(m) = &mut e.data {
                m.elements.pop();
            }
        }
    }
    interaction.markup_preview_live = false;
}

fn push_markup_preview(engine: &mut Engine, interaction: &mut EditorInteractionState) {
    pop_markup_preview(engine, interaction);
    let Some(el) = interaction.markup.preview_element.clone() else {
        return;
    };
    let Some(id) = interaction.markup.target_node else {
        return;
    };
    if let Some(e) = engine.world.graph.get_mut(id) {
        if let NodeData::Markup(m) = &mut e.data {
            m.elements.push(el);
            interaction.markup_preview_live = true;
        }
    }
}

fn measurement_pick(
    engine: &mut Engine,
    session: &mut EditorSession,
    interaction: &mut EditorInteractionState,
    world: [f32; 3],
) {
    if !interaction.measurement_mode {
        return;
    }
    if interaction.measurement_type.is_none() {
        interaction.measurement_type = Some(MeasurementType::Distance);
        interaction.measure = crate::measurement::action_for(MeasurementType::Distance);
    }
    let done = interaction.measure.add_point(Vec3::from_array(world));
    if !done {
        return;
    }
    interaction.measure_label = interaction.measure.label.clone();
    if let Some(&parent) = engine.world.graph.roots().first() {
        if let Some(id) = interaction
            .measure
            .create_annotation_node(&mut engine.world.graph, parent)
        {
            session.dirty = true;
            engine.world.graph.clear_selection();
            engine.world.graph.select(id);
        }
    }
    if let Some(ty) = interaction.measurement_type {
        let label = interaction.measure_label.clone();
        interaction.measure = crate::measurement::action_for(ty);
        interaction.measure_label = label;
    }
}

fn cancel_tool(engine: &mut Engine, interaction: &mut EditorInteractionState) {
    pop_markup_preview(engine, interaction);
    interaction.markup.cancel();
    if interaction.measurement_mode {
        if let Some(ty) = interaction.measurement_type {
            let last = interaction.measure_label.clone();
            interaction.measure = crate::measurement::action_for(ty);
            interaction.measure_label = last;
        }
    }
}
