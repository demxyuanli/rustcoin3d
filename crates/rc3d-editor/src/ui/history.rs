use egui_extras::{Column, TableBuilder};

use crate::commands::EditorCommand;
use crate::ui::i18n::t;
use crate::ui::icons::{self, Icon, PANEL_BTN};
use crate::ui::types::EditorUiContext;

pub(super) fn draw(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    let pal = ui_ctx.ui_theme.palette();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0_f32;
        ui.label(t(loc, "hist.title"));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if icons::icon_button(ui, Icon::Redo, t(loc, "edit.redo"), false, PANEL_BTN, &pal)
                .clicked()
            {
                push(EditorCommand::Redo);
            }
            if icons::icon_button(ui, Icon::Undo, t(loc, "edit.undo"), false, PANEL_BTN, &pal)
                .clicked()
            {
                push(EditorCommand::Undo);
            }
        });
    });
    ui.separator();
    let undo_rows: Vec<_> = ui_ctx.history_undo.iter().enumerate().collect();
    let redo_rows: Vec<_> = ui_ctx.history_redo.iter().enumerate().collect();
    TableBuilder::new(ui)
        .striped(true)
        .resizable(true)
        .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
        .column(Column::auto().at_least(28.0))
        .column(Column::remainder())
        .header(20.0, |mut row| {
            row.col(|ui| {
                ui.strong("#");
            });
            row.col(|ui| {
                ui.strong(t(loc, "hist.title"));
            });
        })
        .body(|mut body| {
            if undo_rows.is_empty() && redo_rows.is_empty() {
                body.row(18.0, |mut row| {
                    row.col(|ui| {
                        ui.weak("-");
                    });
                    row.col(|ui| {
                        ui.weak(t(loc, "hist.empty"));
                    });
                });
                return;
            }
            for (i, desc) in undo_rows {
                let active = i + 1 == ui_ctx.history_undo.len();
                body.row(18.0, |mut row| {
                    row.col(|ui| {
                        ui.label(format!("{}", i + 1));
                    });
                    row.col(|ui| {
                        if ui.selectable_label(active, desc).clicked() {
                            push(EditorCommand::HistoryJump { undo_len: i + 1 });
                        }
                    });
                });
            }
            if !redo_rows.is_empty() {
                body.row(18.0, |mut row| {
                    row.col(|ui| {
                        ui.weak("+");
                    });
                    row.col(|ui| {
                        ui.weak(t(loc, "hist.redo"));
                    });
                });
                for (i, desc) in redo_rows {
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.label(format!("{}", ui_ctx.history_undo.len() + i + 1));
                        });
                        row.col(|ui| {
                            if ui.selectable_label(false, desc).clicked() {
                                push(EditorCommand::HistoryJump {
                                    undo_len: ui_ctx.history_undo.len() + i + 1,
                                });
                            }
                        });
                    });
                }
            }
        });
}
