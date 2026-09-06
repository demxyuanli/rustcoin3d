//! Studio CAD compositor verification against the industrial-look canvas matrix.
//! Invoked with `rc3d-studio --cad-matrix`. Uses the live Engine render path.

use rc3d_core::DisplayMode;
use rc3d_engine_api::Engine;
use rc3d_render::renderer::CadDisplayTier;
use rc3d_render::{CompOp, CompositorGraph};

pub struct MatrixReport {
    pub passed: bool,
    pub text: String,
}

/// One verification row. Rows apply `snapshot` → `tier` → `graph` → `cooldown`
/// in order, then evaluate the engine status line against want/forbid.
struct Row {
    name: &'static str,
    want: &'static [&'static str],
    forbid: &'static [&'static str],
    display: Option<DisplayMode>,
    grid: Option<bool>,
    xray: Option<bool>,
    tier: Option<CadDisplayTier>,
    graph: Option<CompositorGraph>,
    orbit: bool,
    /// Establish the grid/xray snapshot before this row (Preset SSAO+TAA).
    snapshot: bool,
    /// Disable orbit and render 32 idle frames before evaluating (cooldown).
    cooldown: bool,
}

/// Rows run in order. Conditional rows are skipped when the GPU clamps
/// ProductRendering down to IndustrialDisplay.
fn matrix_rows(product_clamped: bool) -> Vec<Row> {
    let mut rows = vec![
        Row {
            name: "Identity + Visualization",
            want: &["CAD: Viz/tier", "Shadow", "Edges"],
            forbid: &["SSAO", "TAA", "HDR", "SSR", "DOF", "Fog", "/comp"],
            display: None,
            grid: None,
            xray: None,
            tier: Some(CadDisplayTier::Visualization),
            graph: Some(CompositorGraph::identity()),
            orbit: false,
            snapshot: false,
            cooldown: false,
        },
        Row {
            name: "Identity + Industrial",
            want: &["CAD: Industrial/tier", "HDR", "SSAO", "TAA", "CG", "Shadow"],
            forbid: &["SSR", "DOF", "Fog", "Edges", "/comp"],
            display: Some(DisplayMode::Shaded),
            grid: None,
            xray: None,
            tier: Some(CadDisplayTier::IndustrialDisplay),
            graph: Some(CompositorGraph::identity()),
            orbit: false,
            snapshot: false,
            cooldown: false,
        },
    ];
    if product_clamped {
        rows.push(Row {
            name: "Identity + Product (GPU clamp to Industrial)",
            want: &["CAD: Industrial/tier", "HDR", "SSAO", "TAA", "CG"],
            forbid: &["SSR", "DOF", "Fog", "CAD: Product/", "/comp"],
            display: None,
            grid: None,
            xray: None,
            tier: Some(CadDisplayTier::ProductRendering),
            graph: Some(CompositorGraph::identity()),
            orbit: false,
            snapshot: false,
            cooldown: false,
        });
    } else {
        rows.push(Row {
            name: "Identity + Product",
            want: &[
                "CAD: Product/tier",
                "HDR",
                "SSAO",
                "TAA",
                "CG",
                "SSR",
                "Fog",
                "DOF",
                "Shadow",
            ],
            forbid: &["Edges", "/comp"],
            display: Some(DisplayMode::Shaded),
            grid: None,
            xray: None,
            tier: Some(CadDisplayTier::ProductRendering),
            graph: Some(CompositorGraph::identity()),
            orbit: false,
            snapshot: false,
            cooldown: false,
        });
    }
    rows.push(Row {
        name: "Orbit Identity Industrial",
        want: &["Viz<-Industrial/tier", "Shadow", "Edges"],
        forbid: &["SSAO", "TAA", "HDR", "SSR", "/comp"],
        display: None,
        grid: None,
        xray: None,
        tier: Some(CadDisplayTier::IndustrialDisplay),
        graph: Some(CompositorGraph::identity()),
        orbit: true,
        snapshot: false,
        cooldown: false,
    });
    // Post-cooldown: judged after 32 idle renders without re-applying state.
    rows.push(Row {
        name: "Cooldown recover Identity Industrial",
        want: &["CAD: Industrial/tier", "HDR", "SSAO", "TAA", "CG"],
        forbid: &["Viz<-", "/comp", "SSR"],
        display: None,
        grid: None,
        xray: None,
        tier: None,
        graph: None,
        orbit: false,
        snapshot: false,
        cooldown: true,
    });
    rows.push(Row {
        name: "Orbit Industrial preset",
        want: &["Viz<-Industrial/comp", "HDR", "SSAO", "TAA", "CG", "Shadow"],
        forbid: &["/tier", "SSR", "DOF"],
        display: None,
        grid: None,
        xray: None,
        tier: Some(CadDisplayTier::IndustrialDisplay),
        graph: Some(CompositorGraph::preset_industrial()),
        orbit: true,
        snapshot: false,
        cooldown: false,
    });
    rows.push(Row {
        name: "Preset Hidden Line",
        want: &["/comp", "HLR"],
        forbid: &["/tier"],
        display: Some(DisplayMode::HiddenLine),
        grid: None,
        xray: None,
        tier: Some(CadDisplayTier::Visualization),
        graph: Some(CompositorGraph::preset_hidden_line()),
        orbit: false,
        snapshot: false,
        cooldown: false,
    });
    rows.push(Row {
        name: "Preset Edges-only",
        want: &["/comp", "Edges"],
        forbid: &["HDR", "SSAO", "TAA", "HLR", "/tier"],
        display: Some(DisplayMode::ShadedWithEdges),
        grid: None,
        xray: None,
        tier: Some(CadDisplayTier::IndustrialDisplay),
        graph: Some(CompositorGraph::preset_edges_only()),
        orbit: false,
        snapshot: false,
        cooldown: false,
    });
    rows.push(Row {
        name: "Preset Product compositor",
        want: &["/comp", "HDR", "SSAO", "TAA", "CG", "SSR", "Fog", "DOF", "Shadow"],
        forbid: &["/tier"],
        display: None,
        grid: None,
        xray: None,
        tier: Some(CadDisplayTier::Visualization),
        graph: Some(CompositorGraph::preset_product()),
        orbit: false,
        snapshot: false,
        cooldown: false,
    });
    rows.push(Row {
        name: "Preset SSAO+TAA (grid/xray forced off)",
        want: &["/comp", "SSAO", "TAA", "HDR"],
        forbid: &["Grid", "XRay", "/tier"],
        display: None,
        grid: Some(false),
        xray: Some(false),
        tier: None,
        graph: Some(CompositorGraph::preset_ssao_taa()),
        orbit: false,
        // Re-establish the grid/xray snapshot (Viz tier + grid + xray on)
        // right before this row, as the original check sequence did.
        snapshot: true,
        cooldown: false,
    });
    rows.push(Row {
        name: "Identity after SSAO+TAA restores snapshot",
        want: &["CAD: Viz/tier", "Grid", "XRay", "Shadow", "Edges"],
        forbid: &["SSAO", "TAA", "HDR", "/comp"],
        display: None,
        grid: Some(true),
        xray: Some(true),
        tier: None,
        graph: Some(CompositorGraph::identity()),
        orbit: false,
        snapshot: false,
        cooldown: false,
    });
    rows.push(Row {
        name: "Beauty Mix Viewer (film only)",
        want: &["CAD: Industrial/tier", "HDR", "SSAO", "TAA", "CG"],
        forbid: &["/comp"],
        display: None,
        grid: None,
        xray: None,
        tier: Some(CadDisplayTier::IndustrialDisplay),
        graph: Some(mix_only()),
        orbit: false,
        snapshot: false,
        cooldown: false,
    });
    rows
}

pub fn run(engine: &mut Engine) -> MatrixReport {
    let mut log = String::new();
    let mut failed = 0u32;

    set_tier(engine, CadDisplayTier::Visualization);
    let product_clamped = {
        set_tier(engine, CadDisplayTier::ProductRendering);
        let req = engine
            .renderer
            .as_ref()
            .map(|r| r.requested_display_tier())
            .unwrap_or(CadDisplayTier::Visualization);
        req != CadDisplayTier::ProductRendering
    };
    log.push_str(&format!(
        "gpu_product_clamped={product_clamped} requested={:?}\n",
        engine
            .renderer
            .as_ref()
            .map(|r| r.requested_display_tier())
    ));

    for mut row in matrix_rows(product_clamped) {
        if row.snapshot {
            // Grid/xray snapshot state: Viz tier + identity + grid + xray on.
            set_tier(engine, CadDisplayTier::Visualization);
            apply(engine, CompositorGraph::identity(), false);
            if let Some(r) = engine.renderer.as_mut() {
                r.set_grid_enabled(true);
            }
            engine.set_xray_mode(true);
        }
        if let Some(tier) = row.tier {
            set_tier(engine, tier);
        }
        if let Some(graph) = row.graph.take() {
            apply(engine, graph, row.orbit);
        } else if row.cooldown {
            set_orbit(engine, false);
            for _ in 0..32 {
                engine.render();
            }
        }
        failed += eval(engine, &mut log, &row);
    }

    set_tier(engine, CadDisplayTier::Visualization);
    let passed = failed == 0;
    log.push_str(&format!(
        "\nresult: {} (failed_rows={failed})\n",
        if passed { "PASS" } else { "FAIL" }
    ));
    MatrixReport { passed, text: log }
}

fn mix_only() -> CompositorGraph {
    let mut g = CompositorGraph::identity();
    let beauty = g
        .nodes
        .iter()
        .find(|n| n.op == CompOp::RenderLayers)
        .map(|n| n.id)
        .expect("beauty");
    let viewer = g
        .nodes
        .iter()
        .find(|n| n.op == CompOp::Viewer)
        .map(|n| n.id)
        .expect("viewer");
    let mix = g.add_node(CompOp::Mix, [260.0, 80.0]);
    g.disconnect_input(viewer, 0);
    g.connect(beauty, mix, 0);
    g.connect(mix, viewer, 0);
    g
}

fn set_tier(engine: &mut Engine, tier: CadDisplayTier) {
    if let Some(r) = engine.renderer.as_mut() {
        r.set_display_tier(tier);
    }
}

fn set_orbit(engine: &mut Engine, orbit: bool) {
    if let Some(r) = engine.renderer.as_mut() {
        r.interaction_active = orbit;
    }
}

fn apply(engine: &mut Engine, graph: CompositorGraph, orbit: bool) {
    engine.compositor = graph;
    set_orbit(engine, orbit);
    engine.render();
}

fn eval(engine: &Engine, log: &mut String, row: &Row) -> u32 {
    let r = engine.renderer.as_ref().expect("renderer");
    let line = r.cad_status_line();
    let display = r.display_mode();
    let grid = r.grid_enabled;
    let xray = r.xray_mode;
    let mut problems: Vec<String> = Vec::new();
    for w in row.want {
        if !line.contains(w) {
            problems.push(format!("missing `{w}`"));
        }
    }
    for f in row.forbid {
        if line.contains(f) {
            problems.push(format!("has forbidden `{f}`"));
        }
    }
    if let Some(d) = row.display {
        if display != d {
            problems.push(format!("display {display:?} != {d:?}"));
        }
    }
    if let Some(g) = row.grid {
        if grid != g {
            problems.push(format!("grid {grid} != {g}"));
        }
    }
    if let Some(x) = row.xray {
        if xray != x {
            problems.push(format!("xray {xray} != {x}"));
        }
    }
    let ok = problems.is_empty();
    if ok {
        log.push_str(&format!("PASS  {} | {line} | {display:?}\n", row.name));
        0
    } else {
        log.push_str(&format!(
            "FAIL  {} | {line} | {display:?} grid={grid} xray={xray} | {}\n",
            row.name,
            problems.join("; ")
        ));
        1
    }
}
