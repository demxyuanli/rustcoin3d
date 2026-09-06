use crate::commands::EditorCommand;
use crate::ui::i18n::t;
use crate::ui::icons::{self, Icon, PANEL_BTN};
use crate::ui::types::{CaseKind, EditorChromeState, EditorUiContext};

/// Assets tab: the case library only (filterable catalog + active-case
/// params/steps). Recent files and import moved back to the File menu.
pub(super) fn draw(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    egui::ScrollArea::vertical()
        .id_salt("assets_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.strong(t(loc, "cases.window"));
            ui.separator();
            case_library(ui, ui_ctx, chrome, push);
        });
}

fn case_library(
    ui: &mut egui::Ui,
    ui_ctx: &EditorUiContext,
    chrome: &mut EditorChromeState,
    push: &mut impl FnMut(EditorCommand),
) {
    let loc = ui_ctx.ui_locale;
    ui.horizontal(|ui| {
        ui.label(t(loc, "cases.filter"));
        ui.add(
            egui::TextEdit::singleline(&mut chrome.cases_filter)
                .desired_width(200.0)
                .hint_text(t(loc, "cases.filter_hint")),
        );
        if icons::icon_button(
            ui,
            Icon::Clear,
            t(loc, "cases.clear_filter"),
            false,
            PANEL_BTN,
            &ui_ctx.ui_theme.palette(),
        )
        .clicked()
        {
            chrome.cases_filter.clear();
        }
    });

    let filter = chrome.cases_filter.to_ascii_lowercase();
    let mut by_cat: Vec<(&str, Vec<&crate::ui::types::CaseListItem>)> = Vec::new();
    for item in &ui_ctx.case_catalog {
        if !filter.is_empty() {
            let id_ok = item.id.to_ascii_lowercase().contains(&filter);
            let title_ok = t(loc, item.title_key)
                .to_ascii_lowercase()
                .contains(&filter);
            let cat_ok = item.category.to_ascii_lowercase().contains(&filter);
            if !(id_ok || title_ok || cat_ok) {
                continue;
            }
        }
        if let Some((_, list)) = by_cat.iter_mut().find(|(c, _)| *c == item.category) {
            list.push(item);
        } else {
            by_cat.push((item.category.as_str(), vec![item]));
        }
    }

    if by_cat.is_empty() {
        ui.weak(t(loc, "cases.empty"));
    }
    for (cat, items) in by_cat {
        ui.collapsing(format!("{cat} ({})", items.len()), |ui| {
            for item in items {
                let kind = match item.kind {
                    CaseKind::Param => t(loc, "case.kind.param"),
                    CaseKind::Process => t(loc, "case.kind.process"),
                    CaseKind::Both => t(loc, "case.kind.both"),
                };
                let title = t(loc, item.title_key);
                let selected = ui_ctx.active_case_id.as_deref() == Some(item.id.as_str());
                let label = format!("{title}  [{kind}]  ({})", item.id);
                if ui.selectable_label(selected, label).clicked() {
                    push(EditorCommand::LoadCase(item.id.clone()));
                }
            }
        });
    }

    if let Some(active_id) = &ui_ctx.active_case_id {
        ui.separator();
        ui.strong(format!("{}: {active_id}", t(loc, "assets.active_case")));
        if !ui_ctx.case_params.is_empty() {
            ui.label(t(loc, "assets.case_params"));
            for p in &ui_ctx.case_params {
                let mut v = p.value;
                let label = t(loc, p.label_key);
                if ui
                    .add(egui::Slider::new(&mut v, p.min..=p.max).text(label))
                    .changed()
                {
                    push(EditorCommand::SetCaseParam { id: p.id.clone(), value: v });
                }
            }
        }
        if !ui_ctx.case_steps.is_empty() {
            ui.label(t(loc, "assets.case_steps"));
            ui.horizontal_wrapped(|ui| {
                for s in &ui_ctx.case_steps {
                    let label = t(loc, s.label_key);
                    let text = if s.active {
                        format!("> {label}")
                    } else {
                        label.to_string()
                    };
                    if ui.button(text).clicked() {
                        push(EditorCommand::RunCaseStep(s.index));
                    }
                }
            });
        }
    } else {
        ui.separator();
        ui.weak(t(loc, "cases.pick_hint"));
    }
}
