//! Top-level egui frame orchestration: caption, menu bar, status bar, docks,
//! the 3D film hole and floating chrome. Layout details live in `docks.rs`;
//! bar/prompt/menu details in `shell.rs`.

use std::collections::VecDeque;

use rc3d_scene::SceneGraph;

use crate::commands::EditorCommand;
use crate::ui::theme::ThemePalette;
use crate::ui::types::{EditorChromeState, EditorUiContext};

use super::docks::{
    draw_bottom_dock_outer, draw_side_dock_outer, paint_scene_overlays,
};
use super::shell::{draw_close_prompt, draw_scene_context_menu, draw_status_bar, flush_panel};

pub(super) fn build_ui(
    ctx: &mut egui::Ui,
    graph: &SceneGraph,
    ui_ctx: &EditorUiContext,
    context_menu_pos: &mut Option<egui::Pos2>,
    chrome: &mut EditorChromeState,
    splash: &mut super::splash::SplashState,
    compositor: &mut super::compositor::CompositorEditor,
    markdown_cache: &mut egui_commonmark::CommonMarkCache,
    q: &mut VecDeque<EditorCommand>,
) {
    // Splash takes over the whole window until the host flips `done`.
    if super::splash::draw_splash(ctx.ctx(), ui_ctx, splash) {
        return;
    }
    let mut push = |c: EditorCommand| q.push_back(c);
    let pal: ThemePalette = ui_ctx.ui_theme.palette();
    let bar_inner = egui::Margin::symmetric(8, 2);
    chrome.side_dock_resizing = false;
    chrome.bottom_dock_resizing = false;

    crate::ui::caption::draw_caption(ctx, chrome, ui_ctx);

    flush_panel(egui::Panel::top("rc3d_menu"), &pal, bar_inner).show(ctx, |ui| {
        super::menus::menu_bar(ui, ui_ctx, chrome, &mut push);
    });

    flush_panel(egui::Panel::bottom("rc3d_status"), &pal, bar_inner)
        .show(ctx, |ui| draw_status_bar(ui, ui_ctx));

    // Base area left by the egui panels (caption/menu/status). Docks and the
    // 3D film are all derived from this one rect so every shared edge matches
    // by construction — egui `Panel` + external drag desyncs its painted rect,
    // cursor advance and PanelState (the handle stopped responding and the
    // docks overlapped the film), so docks are laid out manually now.
    let base = ctx.available_rect_before_wrap();
    let side_dock_left =
        draw_side_dock_outer(ctx, base, graph, ui_ctx, chrome, &pal, &mut push);
    let bottom_dock_top = draw_bottom_dock_outer(
        ctx,
        base,
        side_dock_left,
        ui_ctx,
        chrome,
        compositor,
        markdown_cache,
        &pal,
        &mut push,
    );

    // The 3D film hole: derived from the same `base`, clamped to the dock
    // edges. No `CentralPanel` — its cursor bookkeeping is what drifted from
    // the dock rects and caused the overlap.
    let hole = egui::Rect::from_min_max(
        egui::pos2(base.min.x, base.min.y),
        egui::pos2(
            side_dock_left.unwrap_or(base.max.x),
            bottom_dock_top.unwrap_or(base.max.y),
        ),
    );
    let hole_nonempty = hole.max.x > hole.min.x && hole.max.y > hole.min.y;
    let mut scene_ui = ctx.new_child(
        egui::UiBuilder::new()
            .id_salt("rc3d_scene_hole")
            .max_rect(hole)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    scene_ui.set_clip_rect(hole);
    if hole_nonempty {
        chrome.scene_rect_points = Some([hole.min.x, hole.min.y, hole.width(), hole.height()]);
        paint_scene_overlays(&mut scene_ui, hole, ui_ctx, chrome, &pal, &mut push);
        if ctx.input(|i| i.pointer.button_clicked(egui::PointerButton::Secondary))
            && !ctx.ctx().is_pointer_over_egui()
        {
            if let Some(pos) = ctx.input(|i| i.pointer.interact_pos()) {
                *context_menu_pos = Some(pos);
            }
        }
    } else {
        chrome.scene_rect_points = None;
    }

    if chrome.caption.close_prompt {
        draw_close_prompt(ctx.ctx(), ui_ctx, chrome, &mut push);
    }

    draw_scene_context_menu(
        ctx.ctx(),
        graph,
        ui_ctx,
        context_menu_pos,
        chrome,
        &mut push,
    );
}
