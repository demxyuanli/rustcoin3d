use std::path::Path;

use rc3d_core::math::Vec3;
use rc3d_engine_api::{CameraController, Engine};
use rc3d_scene::SceneGraph;

use crate::commands::EditorCommand;
use crate::context::EditorInteractionState;
use crate::document;

use super::EditorSession;

pub(super) fn apply(
    engine: &mut Engine,
    interaction: &mut EditorInteractionState,
    session: &mut EditorSession,
    cmd: EditorCommand,
) {
    match cmd {
        EditorCommand::NewScene => {
            install_scene(engine, session, interaction, document::blank_scene(), None);
        }
        EditorCommand::OpenScene(path) => match document::load_native_scene(&path) {
            Ok(graph) => {
                install_scene(engine, session, interaction, graph, Some(path));
            }
            Err(e) => log::warn!("open scene failed: {e}"),
        },
        EditorCommand::SaveScene => {
            if let Some(path) = session.document_path.clone() {
                save_bound_scene(engine, session, &path);
            } else {
                log::warn!("save scene: no document path");
            }
        }
        EditorCommand::SaveSceneAs(path) => {
            let path = document::with_json_extension(path);
            if save_bound_scene(engine, session, &path) {
                session.document_path = Some(path);
            }
        }
        EditorCommand::ImportPath(path) => {
            import_path(engine, &path);
            session.dirty = true;
            session.file_dialog_dir = document::dialog_dir_from_path(&path);
        }
        EditorCommand::ExportIvPath(path) => {
            export_scene_json(engine.scene(), &path);
        }
        EditorCommand::Export3dPdf { path, options } => {
            export_3d_pdf(engine.scene(), &path, &options);
            // Remember the export folder so the next export dialog opens
            // here (persisted to prefs as `last_folder`).
            session.file_dialog_dir = document::dialog_dir_from_path(&path);
        }
        EditorCommand::ExportDiagnosticsJsonPath(path) => {
            export_diagnostics(engine, &path);
        }
        EditorCommand::ExportScreenshot(path) => {
            export_screenshot(engine, &path);
        }
        EditorCommand::ExportHiddenLineSvg(path) => {
            if let Err(e) = engine.export_hidden_line_svg(&path) {
                log::warn!("hidden-line SVG export failed: {e}");
            }
        }
        EditorCommand::ExportQuadPack(path) => {
            export_quad_pack(engine, &path);
        }
        EditorCommand::LoadIblHdr(path) => {
            if let Some(r) = engine.renderer.as_mut() {
                r.set_ibl_from_path(path);
            }
        }
        _ => {}
    }
}

fn import_path(engine: &mut Engine, path: &Path) {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "json" => match document::load_native_scene(path) {
            Ok(graph) => engine.load_scene(graph),
            Err(e) => log::warn!("scene load failed: {e}"),
        },
        _ => match engine.import(path) {
            Ok(_) => {
                engine.world.collector.invalidate_mesh_cache();
            }
            Err(e) => log::warn!("import failed: {e}"),
        },
    }
}

fn save_bound_scene(engine: &Engine, session: &mut EditorSession, path: &Path) -> bool {
    match document::save_native_scene(engine.scene(), path) {
        Ok(()) => {
            session.dirty = false;
            session.file_dialog_dir = document::dialog_dir_from_path(path);
            true
        }
        Err(e) => {
            log::warn!("save scene failed: {e}");
            false
        }
    }
}

fn install_scene(
    engine: &mut Engine,
    session: &mut EditorSession,
    interaction: &mut EditorInteractionState,
    graph: SceneGraph,
    path: Option<std::path::PathBuf>,
) {
    engine.load_scene(graph);
    engine.controller = CameraController::new(Vec3::ZERO, 10.0);
    engine.hidden_nodes.clear();
    *interaction = EditorInteractionState::default();
    session.history.clear();
    session.document_path = path;
    if let Some(ref p) = session.document_path {
        session.file_dialog_dir = document::dialog_dir_from_path(p);
    }
    session.dirty = false;
    rc3d_engine_api::sync_gizmo_from_selection(&mut engine.gizmo, &engine.world.graph);
}

fn export_scene_json(graph: &SceneGraph, path: &Path) {
    match serde_json::to_string_pretty(graph) {
        Ok(text) => {
            if let Err(e) = std::fs::write(path, text) {
                log::warn!("export failed: {e}");
            }
        }
        Err(e) => log::warn!("serialize scene failed: {e}"),
    }
}

/// Interactive 3D PDF (U3D) export for the current scene. The document
/// stem becomes the 3D-view title; errors surface on the log.
fn export_3d_pdf(graph: &SceneGraph, path: &Path, options: &rc3d_pdf::PdfOptions) {
    let title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("rc3d scene");
    let with_ext = if path.extension().is_some() {
        path.to_path_buf()
    } else {
        path.with_extension("pdf")
    };
    match rc3d_pdf::export_u3d_pdf_opts(graph, title, options) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(&with_ext, bytes) {
                log::warn!("3D PDF export failed: {e}");
            }
        }
        Err(e) => log::warn!("3D PDF export failed: {e}"),
    }
}

fn export_diagnostics(engine: &Engine, path: &Path) {
    let text = engine
        .renderer
        .as_ref()
        .and_then(|r| r.last_diagnostics())
        .map(|d| format!("{d:#?}"))
        .unwrap_or_else(|| "{}".into());
    if let Err(e) = std::fs::write(path, text) {
        log::warn!("diagnostics export failed: {e}");
    }
}

fn export_screenshot(engine: &mut Engine, path: &Path) {
    let Some(r) = engine.renderer.as_mut() else {
        return;
    };
    let (w, h) = r.surface_size();
    let (w, h, pixels) = r.render_to_image(
        &engine.world.cached_draw_calls,
        &engine.world.graph,
        w.max(1),
        h.max(1),
    );
    save_rgba_png(path, w, h, pixels);
}

fn export_quad_pack(engine: &mut Engine, path: &Path) {
    let (w, h) = engine
        .renderer
        .as_ref()
        .map(|r| r.surface_size())
        .unwrap_or((1280, 720));
    let (w, h, pixels) = engine.render_quad_pack_image(w.max(1), h.max(1));
    save_rgba_png(path, w, h, pixels);
}

fn save_rgba_png(path: &Path, w: u32, h: u32, pixels: Vec<u8>) {
    match image::RgbaImage::from_raw(w, h, pixels) {
        Some(img) => {
            if let Err(e) = img.save(path) {
                log::warn!("image save failed: {e}");
            }
        }
        None => log::warn!("image encode failed"),
    }
}
