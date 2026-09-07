//! Startup splash screen: procedural hero art (left 2/3) and text panel
//! (right 1/3) with a loading progress bar. Drawn entirely with egui shapes
//! so no binary image asset is needed.

use egui::{Color32, CornerRadius, Pos2, Rect, Sense, Stroke, Vec2};

use crate::ui::i18n::t;
use crate::ui::theme::ThemePalette;

/// Loading stages the host advances through while the engine/editor warm up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplashStage {
    /// GPU device / pipelines / swapchain.
    Device,
    /// Scene graph upload and mesh cache warm-up.
    Scene,
    /// Final egui fonts / window reveal.
    Ready,
}

impl SplashStage {
    fn key(self) -> &'static str {
        match self {
            SplashStage::Device => "splash.stage_device",
            SplashStage::Scene => "splash.stage_scene",
            SplashStage::Ready => "splash.stage_ready",
        }
    }
}

/// Host-driven splash state: the current stage and whether loading finished.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplashState {
    pub stage: SplashStage,
    /// Continuous 0..1 bar value written by the host each frame. Discrete
    /// per-stage steps made the bar stall at 75% for most of the show.
    pub progress: f32,
    /// True once the host presented the first full frame after Ready.
    pub done: bool,
    /// True once the splash actually painted a frame (screen size known).
    /// The host reveals the window only after this so it never shows the
    /// un-presented white surface.
    pub painted: bool,
}

impl Default for SplashState {
    fn default() -> Self {
        // Disabled by default: hosts that never advance the splash (examples,
        // panel hosts) must not paint it forever. Studio opts in with
        // `starting()`.
        Self {
            stage: SplashStage::Device,
            progress: 0.0,
            done: true,
            painted: false,
        }
    }
}

impl SplashState {
    /// Splash enabled at its first stage (used by the studio host at startup).
    pub fn starting() -> Self {
        Self {
            stage: SplashStage::Device,
            progress: 0.0,
            done: false,
            painted: false,
        }
    }
}

/// Draws the splash and reports whether the UI still shows it (true = active).
pub(in crate::ui) fn draw_splash(
    ctx: &egui::Context,
    ui_ctx: &crate::ui::types::EditorUiContext,
    splash: &mut SplashState,
) -> bool {
    // Once the host flips `done`, the splash must paint nothing so the real
    // workspace shows through. Painting it unconditionally (then returning
    // `!done`) left an opaque full-screen splash covering the UI forever.
    if splash.done {
        return false;
    }
    let pal: ThemePalette = ui_ctx.ui_theme.palette();
    // Whole-screen painter above the default layer. The first frame can run
    // before egui learned the window size (screen_rect still NaN) — skip
    // painting then; the splash stays active and draws on the next frame.
    let screen = ctx
        .input(|i| i.raw.screen_rect)
        .unwrap_or(Rect::NOTHING);
    if !screen.is_finite() || screen.width() < 40.0 {
        return true;
    }
    splash.painted = true;
    let mut painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("splash_layer"),
    ));
    painter.set_clip_rect(screen);

    // Full-window fill so no un-presented swapchain can flash through.
    painter.rect_filled(screen, CornerRadius::ZERO, pal.solid);

    // Centered card: 2/3 art + 1/3 text.
    let card_w = (screen.width() * 0.62).clamp(560.0, 880.0);
    let card_h = (screen.height() * 0.5).clamp(300.0, 460.0);
    let card = Rect::from_center_size(screen.center(), Vec2::new(card_w, card_h));
    let art_w = card.width() * (2.0 / 3.0);
    let art_rect = Rect::from_min_max(card.min, Pos2::new(card.min.x + art_w, card.max.y));
    let text_rect = Rect::from_min_max(art_rect.right_top(), card.right_bottom());

    painter.rect_filled(
        card,
        CornerRadius::same(10),
        if pal.dark {
            Color32::from_rgb(0x25, 0x25, 0x25)
        } else {
            Color32::from_rgb(0xFF, 0xFF, 0xFF)
        },
    );
    painter.rect_stroke(
        card,
        CornerRadius::same(10),
        Stroke::new(1.0, pal.stroke),
        egui::StrokeKind::Inside,
    );

    draw_hero_art(&painter, art_rect, &pal, ctx.input(|i| i.time));
    draw_text_panel(ctx, text_rect, ui_ctx, splash, &pal);
    !splash.done
}

/// Procedural hero art: layered wireframe mountains + orbit rings + accent cube
/// in the brand palette. Pure shape painting, resolution independent.
fn draw_hero_art(
    painter: &egui::Painter,
    rect: Rect,
    pal: &ThemePalette,
    time: f64,
) -> bool {
    if rect.width() < 40.0 {
        return false;
    }
    let painter = &mut painter.clone();
    painter.set_clip_rect(rect);
    let base_y = rect.bottom();

    // Back haze band.
    painter.rect_filled(
        rect,
        CornerRadius::ZERO,
        if pal.dark {
            Color32::from_rgb(0x1B, 0x2A, 0x38)
        } else {
            Color32::from_rgb(0xDD, 0xEA, 0xF6)
        },
    );

    // Orbit rings.
    let center = Pos2::new(rect.center().x, rect.top() + rect.height() * 0.42);
    let ring = rect.width() * 0.30;
    for (i, f) in [0.62_f32, 0.85, 1.08].iter().enumerate() {
        let r = ring * f;
        let steps = 48;
        let mut pts = Vec::with_capacity(steps + 1);
        for s in 0..=steps {
            let a = (s as f32 / steps as f32) * std::f32::consts::TAU;
            let squash = 0.32_f32;
            let mut p = Pos2::new(
                center.x + a.cos() * r,
                center.y + a.sin() * r * squash,
            );
            // Ellipse wobble so rings read as 3D orbits.
            p.y += (a.sin() * 6.0) * (i as f32 * 0.5);
            pts.push(p);
        }
        let tint = if pal.dark { 0x60 } else { 0x8F };
        painter.add(egui::Shape::line(
            pts,
            Stroke::new(1.2, Color32::from_rgba_unmultiplied(tint, tint, tint, 255)),
        ));
    }

    // Accent cube (the "coin"): floating above the horizon, gentle bob.
    let bob = ((time * 1.6).sin() as f32) * 4.0;
    let cube_c = Pos2::new(center.x, center.y - rect.height() * 0.06 + bob);
    let s = rect.width() * 0.075;
    let accent = pal.accent;
    let accent_dim = Color32::from_rgba_unmultiplied(
        accent.r() / 3,
        accent.g() / 3,
        accent.b() / 3,
        255,
    );
    // Top / left / right faces of an isometric cube.
    let top = [
        Pos2::new(cube_c.x, cube_c.y - s),
        Pos2::new(cube_c.x + s * 0.87, cube_c.y - s * 0.5),
        Pos2::new(cube_c.x, cube_c.y),
        Pos2::new(cube_c.x - s * 0.87, cube_c.y - s * 0.5),
    ];
    let left = [
        top[3],
        top[2],
        Pos2::new(cube_c.x, cube_c.y + s),
        Pos2::new(cube_c.x - s * 0.87, cube_c.y + s * 0.5),
    ];
    let right = [
        top[2],
        top[1],
        Pos2::new(cube_c.x + s * 0.87, cube_c.y + s * 0.5),
        Pos2::new(cube_c.x, cube_c.y + s),
    ];
    painter.add(egui::Shape::convex_polygon(
        top.to_vec(),
        accent,
        Stroke::NONE,
    ));
    painter.add(egui::Shape::convex_polygon(
        left.to_vec(),
        accent_dim,
        Stroke::NONE,
    ));
    painter.add(egui::Shape::convex_polygon(
        right.to_vec(),
        accent.gamma_multiply(0.7),
        Stroke::NONE,
    ));

    // Wireframe mountains: two ridge layers of connected triangles.
    let ridges: [(&[f32], f32); 2] = [(&[0.0, 0.35, 0.12, 0.52, 0.2, 0.68], 0.34), (&[0.0, 0.28, 0.18, 0.4, 0.1, 0.55, 0.3, 0.62], 0.22)];
    for (layer, (peaks, h)) in ridges.iter().enumerate() {
        let tint = if pal.dark {
            Color32::from_rgba_unmultiplied(0x35, 0x50, 0x66, 255)
        } else {
            Color32::from_rgba_unmultiplied(0xA9, 0xC4, 0xDC, 255)
        };
        let fill = if pal.dark {
            Color32::from_rgba_unmultiplied(0x14, 0x20, 0x2B, 255)
        } else {
            Color32::from_rgba_unmultiplied(0xC6, 0xD9, 0xEA, 255)
        };
        let mut poly = Vec::new();
        poly.push(Pos2::new(rect.left(), base_y));
        for (i, ph) in peaks.iter().enumerate() {
            let px = rect.left() + rect.width() * (i as f32 / (peaks.len() - 1) as f32);
            let mut py = base_y - rect.height() * h * ph;
            if layer == 0 {
                py += 8.0; // nearer ridge sits lower
            }
            poly.push(Pos2::new(px, py));
        }
        poly.push(Pos2::new(rect.right(), base_y));
        painter.add(egui::Shape::convex_polygon(poly.clone(), fill, Stroke::NONE));
        painter.add(egui::Shape::line(poly, Stroke::new(1.4, tint)));
    }

    // Foreground ground line.
    painter.line_segment(
        [Pos2::new(rect.left(), base_y), Pos2::new(rect.right(), base_y)],
        Stroke::new(1.0, pal.stroke),
    );
    true
}

fn draw_text_panel(
    ctx: &egui::Context,
    rect: Rect,
    ui_ctx: &crate::ui::types::EditorUiContext,
    splash: &SplashState,
    pal: &ThemePalette,
) {
    let loc = ui_ctx.ui_locale;
    let pad = 24.0_f32;
    let mut ui = egui::Ui::new(
        ctx.clone(),
        egui::Id::new("splash_text"),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("splash_text_layer"),
            ))
            .max_rect(rect)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    ui.set_clip_rect(rect);
    ui.add_space(pad * 0.6);

    // Title.
    ui.add_space(pad);
    ui.heading(
        egui::RichText::new("rustcoin3d")
            .size(30.0)
            .color(pal.text),
    );
    ui.add_space(2.0);
    ui.label(
        egui::RichText::new("Studio")
            .size(16.0)
            .color(pal.text_secondary),
    );
    ui.add_space(6.0);
    ui.label(
        egui::RichText::new(t(loc, "splash.tagline"))
            .size(12.0)
            .color(pal.text_secondary),
    );

    // Progress anchored to the bottom.
    ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
        ui.add_space(pad * 0.8);
        let bar_h = 5.0_f32;
        let (bar_rect, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), bar_h),
            Sense::hover(),
        );
        ui.painter().rect_filled(bar_rect, CornerRadius::same(2), pal.card);
        let p = splash.progress.clamp(0.0, 1.0);
        if p > 0.001 {
            let fill = Rect::from_min_size(
                bar_rect.min,
                Vec2::new(bar_rect.width() * p, bar_h),
            );
            ui.painter().rect_filled(fill, CornerRadius::same(2), pal.accent);
        }
        ui.add_space(6.0);
        let pct = (p * 100.0).round() as u32;
        ui.label(
            egui::RichText::new(format!("{}  {}%", t(loc, splash.stage.key()), pct))
                .size(11.0)
                .color(pal.text_secondary),
        );
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new(t(loc, "splash.copyright"))
                .size(10.0)
                .color(pal.text_secondary.gamma_multiply(0.7)),
        );
    });
}
