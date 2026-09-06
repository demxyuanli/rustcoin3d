//! Side/bottom docks, dock resize handles, hierarchy split and the floating
//! tool strip. All dock geometry derives from the caller's `base` rect so
//! shared edges match by construction.

use rc3d_scene::SceneGraph;

use crate::commands::EditorCommand;
use crate::ui::i18n::t;
use crate::ui::theme::{chrome_panel_frame, floating_overlay_frame, ThemePalette};
use crate::ui::types::{BottomTab, EditorChromeState, EditorUiContext, SideTab};

const STUDIO_HELP_MD: &str = include_str!("../../assets/studio_help.md");

pub(super) const SIDE_DOCK_ID: &str = "side_dock";
pub(super) const BOTTOM_DOCK_ID: &str = "bottom_dock";
pub(super) const DOCK_RESIZE_GRAB: f32 = 5.0_f32;

/// Chrome owns dock size. egui `Panel` resize is disabled: its `PanelState` is skipped
/// during drag, and `response.rect` follows content `min_rect` (not the allocated
/// outer size), so writing that width back made the right dock snap after release.
fn apply_side_dock_resize(ui: &egui::Ui, width: &mut f32) {
    let resize_id = egui::Id::new(SIDE_DOCK_ID).with("dock_resize");
    let available = ui.available_rect_before_wrap();

    let max_w = (available.width() - 80.0_f32).max(200.0_f32);
    *width = width.clamp(200.0_f32, max_w);
    if let Some(resp) = ui.ctx().read_response(resize_id) {
        if resp.dragged() || resp.drag_stopped() {
            if let Some(pos) = resp.interact_pointer_pos() {
                *width = (available.max.x - pos.x).clamp(200.0_f32, max_w);
                ui.ctx().request_repaint();
            }
        }
    }
}

fn register_side_dock_resize(ui: &egui::Ui, panel_rect: egui::Rect, pal: &ThemePalette) -> bool {
    let resize_id = egui::Id::new(SIDE_DOCK_ID).with("dock_resize");
    let g = DOCK_RESIZE_GRAB;
    let handle = egui::Rect::from_x_y_ranges(
        (panel_rect.left() - g)..=(panel_rect.left() + g),
        panel_rect.y_range(),
    );
    let resp = ui.interact(handle, resize_id, egui::Sense::click_and_drag());
    if resp.hovered() || resp.dragged() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        paint_dock_hint(
            ui,
            true,
            panel_rect.left() - 1.0,
            2.0,
            panel_rect.y_range(),
            resp.dragged(),
            pal,
        );
    }
    if resp.dragged() {
        ui.ctx().request_repaint();
    }
    resp.dragged()
}

fn apply_bottom_dock_resize(ui: &egui::Ui, height: &mut f32) {
    let resize_id = egui::Id::new(BOTTOM_DOCK_ID).with("dock_resize");
    let available = ui.available_rect_before_wrap();
    let max_h = (available.height() - 80.0_f32).max(120.0_f32);
    *height = height.clamp(120.0_f32, max_h);
    if let Some(resp) = ui.ctx().read_response(resize_id) {
        if resp.dragged() || resp.drag_stopped() {
            if let Some(pos) = resp.interact_pointer_pos() {
                *height = (available.max.y - pos.y).clamp(120.0_f32, max_h);
                ui.ctx().request_repaint();
            }
        }
    }
}

fn register_bottom_dock_resize(ui: &egui::Ui, panel_rect: egui::Rect, pal: &ThemePalette) -> bool {
    let resize_id = egui::Id::new(BOTTOM_DOCK_ID).with("dock_resize");
    let g = DOCK_RESIZE_GRAB;
    let handle = egui::Rect::from_x_y_ranges(
        panel_rect.x_range(),
        (panel_rect.top() - g)..=(panel_rect.top() + g),
    );
    let resp = ui.interact(handle, resize_id, egui::Sense::click_and_drag());
    if resp.hovered() || resp.dragged() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
        paint_dock_hint(
            ui,
            false,
            panel_rect.top() - 1.0,
            2.0,
            panel_rect.x_range(),
            resp.dragged(),
            pal,
        );
    }
    if resp.dragged() {
        ui.ctx().request_repaint();
    }
    resp.dragged()
}

/// Foreground edge highlight while hovering/dragging a dock resize handle.
/// `vertical=true` draws a thin column at `line_pos` spanning `span` (side dock);
/// otherwise a thin row at `line_pos` spanning `span` (bottom dock).
pub(super) fn paint_dock_hint(
    ui: &egui::Ui,
    vertical: bool,
    line_pos: f32,
    thickness: f32,
    span: egui::Rangef,
    dragging: bool,
    pal: &ThemePalette,
) {
    if span.max < span.min {
        return;
    }
    let color = if dragging {
        pal.accent
    } else {
        pal.text_secondary.gamma_multiply(0.55)
    };
    let rect = if vertical {
        egui::Rect::from_x_y_ranges(line_pos..=(line_pos + thickness), span.min..=span.max)
    } else {
        egui::Rect::from_x_y_ranges(span.min..=span.max, line_pos..=(line_pos + thickness))
    };
    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("rc3d_dock_resize_hint"),
    ));
    painter.rect_filled(rect, egui::CornerRadius::ZERO, color);
}

// ---------------------------------------------------------------------------
// Scene overlays (inside the 3D film hole)
// ---------------------------------------------------------------------------

/// Overlays drawn inside the 3D film hole: viewport splitter hints, nav cube
/// and the floating tool strip.
pub(super) fn paint_scene_overlays(
    scene_ui: &mut egui::Ui,
    hole: egui::Rect,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    pal: &ThemePalette,
    push: &mut impl FnMut(EditorCommand),
) {
    paint_view_splitter_hints(scene_ui, hole, ui_ctx, pal);
    crate::ui::nav_cube::draw_nav_cube(scene_ui, hole, ui_ctx, chrome, push);
    draw_floating_tool_strip(scene_ui.ctx(), hole, chrome, ui_ctx, pal, push);
}

fn paint_view_splitter_hints(
    ui: &egui::Ui,
    scene: egui::Rect,
    ui_ctx: &EditorUiContext,
    pal: &ThemePalette,
) {
    use rc3d_render::viewport::{LayoutMode, ViewportSplitAxis};

    let hover = ui_ctx.viewport_split_drag.or(ui_ctx.viewport_split_hover);
    let Some(axis) = hover else {
        return;
    };
    if scene.width() < 2.0 || scene.height() < 2.0 {
        return;
    }
    let dragging = ui_ctx.viewport_split_drag.is_some();
    match ui_ctx.layout_mode {
        LayoutMode::Single => {}
        LayoutMode::Quad => {
            if matches!(axis, ViewportSplitAxis::HorizontalFraction) {
                paint_dock_hint(
                    ui,
                    true,
                    scene.left() + scene.width() * ui_ctx.viewport_h_split - 1.0,
                    2.0,
                    scene.y_range(),
                    dragging,
                    pal,
                );
            } else {
                paint_dock_hint(
                    ui,
                    false,
                    scene.top() + scene.height() * ui_ctx.viewport_v_split - 1.0,
                    2.0,
                    scene.x_range(),
                    dragging,
                    pal,
                );
            }
        }
        LayoutMode::LeftRight => {
            paint_dock_hint(
                ui,
                true,
                scene.left() + scene.width() * ui_ctx.viewport_h_split - 1.0,
                2.0,
                scene.y_range(),
                dragging,
                pal,
            );
        }
        LayoutMode::TopBottom => {
            paint_dock_hint(
                ui,
                false,
                scene.top() + scene.height() * ui_ctx.viewport_v_split - 1.0,
                2.0,
                scene.x_range(),
                dragging,
                pal,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Dock layout (called from `build_ui`)
// ---------------------------------------------------------------------------

/// Lay out the side dock inside `base`; returns its left edge when visible.
pub(super) fn draw_side_dock_outer(
    ui: &mut egui::Ui,
    base: egui::Rect,
    graph: &SceneGraph,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    pal: &ThemePalette,
    push: &mut impl FnMut(EditorCommand),
) -> Option<f32> {
    let side_inner = egui::Margin::same(8);
    let tab = chrome.side_tab?;
    let mut side_w = chrome.side_dock_width.unwrap_or(280.0);
    apply_side_dock_resize(ui, &mut side_w);
    chrome.side_dock_width = Some(side_w);
    let side_rect = egui::Rect::from_min_max(
        egui::pos2((base.max.x - side_w).max(base.min.x), base.min.y),
        egui::pos2(base.max.x, base.max.y),
    );
    let mut side_ui = ui.new_child(
        egui::UiBuilder::new()
            .id_salt(SIDE_DOCK_ID)
            .max_rect(side_rect)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    side_ui.set_clip_rect(side_rect);
    chrome_panel_frame(pal, side_inner).show(&mut side_ui, |ui| {
        ui.set_min_width(ui.available_width());
        draw_side_dock(ui, graph, ui_ctx, chrome, tab, push);
    });
    chrome.side_dock_resizing = register_side_dock_resize(ui, side_rect, pal);
    Some(side_rect.left())
}

/// Lay out the bottom dock inside `base` (stopping at `side_dock_left`);
/// returns its top edge when visible.
#[allow(clippy::too_many_arguments)]
pub(super) fn draw_bottom_dock_outer(
    ui: &mut egui::Ui,
    base: egui::Rect,
    side_dock_left: Option<f32>,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    compositor: &mut super::compositor::CompositorEditor,
    markdown_cache: &mut egui_commonmark::CommonMarkCache,
    pal: &ThemePalette,
    push: &mut impl FnMut(EditorCommand),
) -> Option<f32> {
    let side_inner = egui::Margin::same(8);
    let tab = chrome.bottom_tab?;
    let mut bottom_h = chrome.bottom_dock_height.unwrap_or(220.0);
    apply_bottom_dock_resize(ui, &mut bottom_h);
    chrome.bottom_dock_height = Some(bottom_h);
    // The bottom dock stops at the side dock's left edge, mirroring egui's
    // right-panel behavior.
    let bottom_max_x = side_dock_left.unwrap_or(base.max.x);
    let bottom_rect = egui::Rect::from_min_max(
        egui::pos2(base.min.x, (base.max.y - bottom_h).max(base.min.y)),
        egui::pos2(bottom_max_x, base.max.y),
    );
    let mut bottom_ui = ui.new_child(
        egui::UiBuilder::new()
            .id_salt(BOTTOM_DOCK_ID)
            .max_rect(bottom_rect)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    bottom_ui.set_clip_rect(bottom_rect);
    chrome_panel_frame(pal, side_inner).show(&mut bottom_ui, |ui| {
        ui.set_min_height(ui.available_height());
        draw_bottom_dock(ui, chrome, compositor, tab, ui_ctx, markdown_cache, push);
    });
    chrome.bottom_dock_resizing = register_bottom_dock_resize(ui, bottom_rect, pal);
    Some(bottom_rect.top())
}

// ---------------------------------------------------------------------------
// Dock contents
// ---------------------------------------------------------------------------

/// Shared tab row for the side/bottom dock headers: horizontal tab strip with
/// an RTL close button. Picking writes `Some(tab)` into `tab_slot`; closing
/// writes `None`.
fn dock_tab_bar<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    tab_slot: &mut Option<T>,
    tabs: &[(T, &'static str)],
    icon_of: fn(T) -> super::icons::Icon,
    salt: &str,
    loc: crate::ui::i18n::UiLocale,
    pal: &ThemePalette,
) {
    let selected = tab_slot.unwrap();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0_f32;
        let close_w = super::icons::PANEL_BTN + ui.spacing().item_spacing.x;
        let tabs_w = (ui.available_width() - close_w).max(0.0_f32);
        egui::ScrollArea::horizontal()
            .id_salt(salt)
            .max_width(tabs_w)
            .auto_shrink([false, true])
            .max_height(super::icons::PANEL_BTN + 8.0_f32)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0_f32;
                    for (id, key) in tabs {
                        if super::icons::tab_button(
                            ui,
                            icon_of(*id),
                            t(loc, key),
                            *id == selected,
                            pal,
                        )
                        .clicked()
                        {
                            *tab_slot = Some(*id);
                        }
                    }
                });
            });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if super::icons::icon_button(
                ui,
                super::icons::Icon::Close,
                t(loc, "panel.close"),
                false,
                super::icons::PANEL_BTN,
                pal,
            )
            .clicked()
            {
                *tab_slot = None;
            }
        });
    });
}

fn draw_side_dock(
    ui: &mut egui::Ui,
    graph: &SceneGraph,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    tab: SideTab,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    let pal = ui_ctx.ui_theme.palette();
    const SIDE_TABS: [(SideTab, &str); 4] = [
        (SideTab::Hierarchy, "panel.hierarchy"),
        (SideTab::Render, "panel.render"),
        (SideTab::History, "panel.history"),
        (SideTab::Assets, "panel.assets"),
    ];
    dock_tab_bar(
        ui,
        &mut chrome.side_tab,
        &SIDE_TABS,
        super::icons::side_tab_icon,
        "side_dock_tabs",
        loc,
        &pal,
    );
    ui.separator();
    match chrome.side_tab.unwrap_or(tab) {
        SideTab::Hierarchy => {
            draw_hierarchy_split(ui, graph, ui_ctx, chrome, push, &pal);
        }
        SideTab::Render => {
            super::inspector::draw_render_panel(ui, ui_ctx, push);
        }
        SideTab::History => {
            super::history::draw(ui, ui_ctx, push);
        }
        SideTab::Assets => {
            super::assets::draw(ui, ui_ctx, chrome, push);
        }
    }
}

fn draw_bottom_dock(
    ui: &mut egui::Ui,
    chrome: &mut EditorChromeState,
    compositor: &mut super::compositor::CompositorEditor,
    tab: BottomTab,
    ui_ctx: &EditorUiContext,
    markdown_cache: &mut egui_commonmark::CommonMarkCache,
    _push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    let pal = ui_ctx.ui_theme.palette();
    const BOTTOM_TABS: [(BottomTab, &str); 2] = [
        (BottomTab::Document, "panel.document"),
        (BottomTab::Compositor, "panel.compositor"),
    ];
    dock_tab_bar(
        ui,
        &mut chrome.bottom_tab,
        &BOTTOM_TABS,
        super::icons::bottom_tab_icon,
        "bottom_dock_tabs",
        loc,
        &pal,
    );
    ui.separator();
    match chrome.bottom_tab.unwrap_or(tab) {
        BottomTab::Document => {
            let avail_h = ui.available_height().max(40.0_f32);
            let avail_w = ui.available_width();
            egui::ScrollArea::both()
                .id_salt("studio_help_md")
                .auto_shrink([false, false])
                .max_height(avail_h)
                .show(ui, |ui| {
                    ui.set_min_width(avail_w);
                    ui.set_max_width(avail_w);
                    egui_commonmark::CommonMarkViewer::new().show(
                        ui,
                        markdown_cache,
                        STUDIO_HELP_MD,
                    );
                });
        }
        BottomTab::Compositor => {
            super::compositor::draw_panel(ui, compositor, loc);
        }
    }
}

/// Hierarchy tab: scene tree on top, selected-node inspector below, separated
/// by a draggable divider. The ratio persists across sessions.
fn draw_hierarchy_split(
    ui: &mut egui::Ui,
    graph: &SceneGraph,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
    pal: &ThemePalette,
) {
    let total_h = ui.available_height();
    // `inspector_ratio` is the inspector's share; the tree gets the rest.
    let ratio = chrome.inspector_ratio.unwrap_or(0.4).clamp(0.2, 0.8);
    let top_h = (total_h * (1.0 - ratio)).clamp(80.0, (total_h - 96.0).max(80.0));

    // Region bounds. Do NOT take `ui.cursor().max` here: in a top-down
    // layout the cursor's max side is infinite, and an infinite max_rect
    // makes the inner ScrollArea believe it never needs scrollbars.
    let avail = ui.available_rect_before_wrap();
    let content_top = avail.min.y;
    let content_bottom = avail.max.y;
    let left = avail.min.x;
    let right = avail.max.x;

    // --- Top: scene tree -------------------------------------------------
    let tree_rect = egui::Rect::from_min_max(
        egui::pos2(left, content_top),
        egui::pos2(right, content_top + top_h),
    );
    let mut tree_ui = ui.new_child(
        egui::UiBuilder::new()
            .id_salt("side_hierarchy_tree")
            .max_rect(tree_rect)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    tree_ui.set_clip_rect(tree_rect);
    super::hierarchy::draw_hierarchy(&mut tree_ui, graph, ui_ctx, chrome, push);
    ui.expand_to_include_rect(tree_rect);

    // --- Divider: drag to resize ------------------------------------------
    let g = DOCK_RESIZE_GRAB;
    let divider_id = ui.make_persistent_id("hierarchy_inspector_divider");
    let divider_y = tree_rect.bottom();
    let handle =
        egui::Rect::from_x_y_ranges(tree_rect.x_range(), (divider_y - g)..=(divider_y + g));
    let resp = ui.interact(handle, divider_id, egui::Sense::click_and_drag());
    if resp.hovered() || resp.dragged() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
    }
    if resp.dragged() {
        if let Some(pos) = resp.interact_pointer_pos() {
            let new_ratio = ((total_h - (pos.y - tree_rect.top())) / total_h).clamp(0.2, 0.8);
            chrome.inspector_ratio = Some(new_ratio);
        }
        ui.ctx().request_repaint();
    }
    let drag_active = resp.dragged();
    paint_dock_hint(
        ui,
        false,
        divider_y - 1.0,
        2.0,
        tree_rect.x_range(),
        drag_active,
        pal,
    );

    // --- Bottom: inspector ------------------------------------------------
    let insp_rect = egui::Rect::from_min_max(
        egui::pos2(left, divider_y),
        egui::pos2(right, content_bottom),
    );
    ui.expand_to_include_rect(insp_rect);
    let mut insp_ui = ui.new_child(
        egui::UiBuilder::new()
            .id_salt("side_hierarchy_inspector")
            .max_rect(insp_rect)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    insp_ui.set_clip_rect(insp_rect);
    super::inspector::draw_inspector_tab(&mut insp_ui, graph, ui_ctx, chrome, push);
}

// ---------------------------------------------------------------------------
// Floating tool strip
// ---------------------------------------------------------------------------

pub(super) fn draw_floating_tool_strip(
    ctx: &egui::Context,
    scene: egui::Rect,
    chrome: &mut EditorChromeState,
    ui_ctx: &EditorUiContext,
    pal: &ThemePalette,
    push: &mut impl FnMut(EditorCommand),
) {
    const MARGIN: f32 = 10.0_f32;
    let strip_size = egui::vec2(40.0_f32, 360.0_f32);
    // Keep a 10px inset from the 3D film edges (default: left + top).
    let confine = egui::Rect::from_min_max(
        scene.min + egui::vec2(MARGIN, MARGIN),
        (scene.max - egui::vec2(MARGIN, MARGIN)).max(scene.min + egui::vec2(MARGIN, MARGIN)),
    );
    let default_pos = confine.min;
    let origin = chrome
        .tool_strip_pos
        .map(|[x, y]| clamp_strip_pos(egui::pos2(x, y), confine, strip_size))
        .unwrap_or(default_pos);
    let strip = egui::Area::new(egui::Id::new("rc3d_tool_strip"))
        .order(egui::Order::Foreground)
        .movable(true)
        .current_pos(origin)
        .default_size(strip_size)
        .constrain_to(confine)
        .show(ctx, |ui| {
            floating_overlay_frame(pal, egui::Margin::symmetric(6, 6)).show(ui, |ui| {
                super::tool_strip::draw(ui, ui_ctx, chrome, push);
            });
        });
    let r = strip.response.rect;
    let clamped = clamp_strip_pos(r.min, confine, r.size());
    chrome.tool_strip_pos = Some([clamped.x, clamped.y]);
    chrome.tool_strip_rect_points = Some([r.min.x, r.min.y, r.width(), r.height()]);
}

fn clamp_strip_pos(pos: egui::Pos2, confine: egui::Rect, size: egui::Vec2) -> egui::Pos2 {
    let max_x = (confine.max.x - size.x).max(confine.min.x);
    let max_y = (confine.max.y - size.y).max(confine.min.y);
    egui::pos2(
        pos.x.clamp(confine.min.x, max_x),
        pos.y.clamp(confine.min.y, max_y),
    )
}
